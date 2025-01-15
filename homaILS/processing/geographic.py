import numpy as np
from pyproj import Proj

# WGS84 constants
WGS84_A = 6378137.0        # major axis
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2*WGS84_F - WGS84_F**2  # eccentricity squared


def geodetic_to_enu(lon_deg, lat_deg, h, lon0_deg, lat0_deg, h0):
    """
    Convert WGS84 geodetic coordinates (longitude, latitude, altitude) to
    local ENU coordinates (East, North, Up), given a reference point.
    
    All angles in degrees, altitudes in meters.
    """
    def geodetic_to_ecef(lon_deg, lat_deg, h):
        """
        Convert WGS84 geodetic coordinates (longitude, latitude, altitude) to
        ECEF coordinates (X, Y, Z).
        - Inputs:
            - lon_deg : float, Longitude in decimal degrees.
            - lat_deg : float, Latitude in decimal degrees.
            - h : float, Altitude in meters.
        - Outputs:
            - X : float, ECEF X coordinate in meters.
            - Y : float, ECEF Y coordinate in meters.
            - Z : float, ECEF Z coordinate in meters.
        """
        # Convert to radians
        lon = np.radians(lon_deg)
        lat = np.radians(lat_deg)

        # Compute prime vertical radius of curvature
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * (np.sin(lat) ** 2))

        # Compute ECEF coordinates
        X = (N + h) * np.cos(lat) * np.cos(lon)
        Y = (N + h) * np.cos(lat) * np.sin(lon)
        Z = (N * (1 - WGS84_E2) + h) * np.sin(lat)
        return X, Y, Z

    def ecef_to_enu(X, Y, Z, lon0_deg, lat0_deg, h0):
        """
        Convert ECEF coordinates (X, Y, Z) to local ENU coordinates (E, N, U),
        given a reference point in geodetic (lon0_deg, lat0_deg, h0).
        - Inputs:
            - X : float, ECEF X coordinate in meters.
            - Y : float, ECEF Y coordinate in meters.
            - Z : float, ECEF Z coordinate in meters.
            - lon0_deg : float, Reference longitude in decimal degrees.
            - lat0_deg : float, Reference latitude in decimal degrees.
            - h0 : float, Reference altitude in meters.
        - Outputs:
            - E : float, Local ENU East coordinate in meters.
            - N : float, Local ENU North coordinate in meters.
            - U : float, Local ENU Up coordinate in meters
        """
        # 1) Convert reference geodetic to ECEF
        X0, Y0, Z0 = geodetic_to_ecef(lon0_deg, lat0_deg, h0)

        # 2) Form rotation matrix from ECEF to ENU
        #    Note lat0, lon0 must be in radians:
        lon0 = np.radians(lon0_deg)
        lat0 = np.radians(lat0_deg)

        # Matrix to rotate ECEF to ENU
        #  -- East  = -sin(lon0) * dX + cos(lon0) * dY
        #  -- North = -sin(lat0)*cos(lon0)*dX - sin(lat0)*sin(lon0)*dY + cos(lat0)*dZ
        #  -- Up    =  cos(lat0)*cos(lon0)*dX + cos(lat0)*sin(lon0)*dY + sin(lat0)*dZ
        #
        # We'll assemble it systematically:

        R = np.array([
            [-np.sin(lon0),              np.cos(lon0),              0],
            [-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)],
            [ np.cos(lat0)*np.cos(lon0),  np.cos(lat0)*np.sin(lon0), np.sin(lat0)]
        ])

        # 3) Subtract reference point
        dX = X - X0
        dY = Y - Y0
        dZ = Z - Z0
        dXYZ = np.array([dX, dY, dZ])

        # 4) Apply rotation
        enu = R.dot(dXYZ)
        E, N, U = enu[0], enu[1], enu[2]

        return E, N, U
    
    X, Y, Z = geodetic_to_ecef(lon_deg, lat_deg, h)
    E, N, U = ecef_to_enu(X, Y, Z, lon0_deg, lat0_deg, h0)
    return E, N, U


def geodetic_to_utm(lon_deg, lat_deg, lon0_deg, lat0_deg, zone):
    """
    Convert WGS84 geodetic coordinates (longitude, latitude) to
    UTM coordinates (E, N), given a reference point.
    - Inputs:
        - lon_deg : float, Longitude in decimal degrees.
        - lat_deg : float, Latitude in decimal degrees.
        - lon0_deg : float, Reference longitude in decimal degrees.
        - lat0_deg : float, Reference latitude in decimal degrees.
    - Outputs:
        - E : float, UTM East coordinate in meters.
        - N : float, UTM North coordinate in meters
    """
    # Coordinate projection for UTM zone
    utm_proj = Proj(proj="utm", zone=zone, ellps="WGS84", south=False)

    # Convert to UTM
    E, N = utm_proj(lon_deg, lat_deg)
    E0, N0 = utm_proj(lon0_deg, lat0_deg)
    return E - E0, N - N0