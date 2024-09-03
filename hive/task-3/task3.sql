ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-contrib.jar;
ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-serde.jar;

USE kuznetsovni;

SELECT IPRegions.mask AS region, SUM(IF(gender = "male", 1, 0)) AS male, SUM(IF(gender = "female", 1, 0)) AS female
FROM Logs, Users, IPRegions
WHERE Logs.ip = Users.ip AND Users.ip = IPRegions.ip
GROUP BY IPRegions.mask;

