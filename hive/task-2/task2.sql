ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-contrib.jar;
ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-serde.jar;

USE kuznetsovni;

SELECT request_time, COUNT(*) AS cnt FROM Logs
GROUP BY request_time
ORDER BY cnt DESC;

