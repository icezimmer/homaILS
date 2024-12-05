#Overview

This file details the data collection campaign based on Bluetooth 5.1 devices estimating AoA and RSS values of Bluetooth tags and a set of anchor nodes deployed in the environment.  We deployed 4 anchor nodes and we use 4 Bluetooth tags to collect data. We reproduce four application scenarios covering typical use-cases for indoor localization: calibration, static, mobility and proximity scenario all including an accurate ground truth (GT) annotation.

#How to cite
A Bluetooth 5.1 Dataset Based on Angle of Arrival and RSS for Indoor Localization, by M. Girolami, F. Furfari, P. Barsocchi, F. Mavilia ---paper under review

#Hardware settings

##Bluetooth tag:

    - C209 tags are equipped with the NINA-B406 BLE module, supporting EddyStone beacon's format on 3 Bluetooth channels (37, 38 and 39).

    - advertisement frequency: 50 Hz

    - power of transmission: 0 dBm

    - ID 8401, MAC address: CCF9579D6F3A
    - ID 8403, MAC address: CCF9579DCD51
    - ID 8404, MAC address: 6C1DEBA42212
    - ID 8402, MAC address: CCF9579D6FC1

##Anchor nodes
All anchors are positioned on a tripod 2.30m from the ground
Anchors are perpendicular w.r.t. the floor (x,y,z):

    - ID anchor 6501, West  (0.00, 3.00, 2.30)
    - ID anchor 6502, South (6.00, 0.00, 2.30)
    - ID anchor 6503, East  (12.00, 3.00, 2.30)
    - ID anchor 6504, North (6.00, 6.00, 2.30)

#Application Scenarios

##Calibration: the goal is to collect data from 4 anchors and 1 tag mounted on a tripod and positioned in 119 different locations of the testing environment.
    - 119 tested locations (see grid_details.txt), 1 minute for each location
    - tag 8401 positioned on top of a tripod oriented toward East; Z(tag 8401) = 1.10m



##Static: the goal is to collect data from 4 anchors and 1 tag held by a person resting in 36 different locations.
    - 36 tested locations (see grid_details.txt), 1 minute for each location
    - tag 8401 held by a person around the neck. Z(tag 8401) = 1.13m
    - 4 orientation of the person: North, South, East, West


##Mobility: the goal is to collect data from 4 anchors and a person holding the tag while moving along 3 different paths.
    - tag 8401 held by a person around the neck; Z(tag 8401) = 1.13m
    - details reported in the paper

##Proximity: the goal is to collect data from 4 anchors and a several people moving in close proximity.
    - multiple tags employed, details in the paper
    - Z(tag 8401) = 1.13m; Z(tag 8402) = 1.16m; Z(tag 8403) = 1.12m; Z(tag 8404) = 1.16m;
