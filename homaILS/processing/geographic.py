import numpy as np

# WGS84 constants
WGS84_A = 6378137.0        # major axis
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2*WGS84_F - WGS84_F**2  # eccentricity squared


def geodetic_to_ecef(lat_deg, lon_deg, h):
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to
    ECEF coordinates (X, Y, Z).
    
    lat_deg, lon_deg in degrees
    h in meters
    returns X, Y, Z in meters (float)
    """
    # Convert to radians
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # Compute prime vertical radius of curvature
    N = WGS84_A / np.sqrt(1 - WGS84_E2 * (np.sin(lat) ** 2))

    # Compute ECEF coordinates
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = (N * (1 - WGS84_E2) + h) * np.sin(lat)
    return X, Y, Z


def ecef_to_enu(X, Y, Z, lat0_deg, lon0_deg, h0):
    """
    Convert ECEF coordinates (X, Y, Z) to local ENU coordinates (E, N, U),
    given a reference point in geodetic (lat0_deg, lon0_deg, h0).
    
    All angles in degrees, altitudes in meters, ECEF coords in meters.
    """
    # 1) Convert reference geodetic to ECEF
    X0, Y0, Z0 = geodetic_to_ecef(lat0_deg, lon0_deg, h0)

    # 2) Form rotation matrix from ECEF to ENU
    #    Note lat0, lon0 must be in radians:
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)

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


def geodetic_to_enu(lat_deg, lon_deg, h, lat0_deg, lon0_deg, h0):
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to
    local ENU coordinates (East, North, Up), given a reference point.
    
    All angles in degrees, altitudes in meters.
    """
    X, Y, Z = geodetic_to_ecef(lat_deg, lon_deg, h)
    E, N, U = ecef_to_enu(X, Y, Z, lat0_deg, lon0_deg, h0)
    return E, N, U