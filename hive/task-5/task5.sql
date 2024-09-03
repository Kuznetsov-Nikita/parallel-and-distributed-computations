ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-contrib.jar;
ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-serde.jar;

USE kuznetsovni;

SELECT TRANSFORM(ip, request_time, http_request, page_size, status_code, information)
USING "sed -r 's|.ru/|.com/|'" AS ip, request_time, http_request, page_size, status_code, information
FROM Logs
LIMIT 10;

