ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-contrib.jar;
ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-exec.jar;

ADD JAR Reverse.jar;

USE kuznetsovni;

CREATE TEMPORARY FUNCTION reverse AS 'ru.mipt.ReverseUDF';

SELECT reverse(ip) AS ip
FROM Subnets
LIMIT 10;

