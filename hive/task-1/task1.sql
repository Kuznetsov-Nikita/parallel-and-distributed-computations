ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-contrib.jar;
ADD JAR /opt/cloudera/parcels/CDH/lib/hive/lib/hive-serde.jar;

USE kuznetsovni;

DROP TABLE IF EXISTS LogsTmp;

CREATE TEMPORARY TABLE LogsTmp (
    ip           STRING,
    request_time STRING,
    http_request STRING,
    page_size    SMALLINT,
    status_code  SMALLINT,
    information  STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
WITH SERDEPROPERTIES (
    "input.regex" = '^(\\S*)\\t\\t\\t(\\d{8})\\S*\\t(\\S*)\\t(\\d*)\\t(\\d*)\\t(\\S*).*$'
)
STORED AS TEXTFILE
LOCATION '/data/user_logs/user_logs_M';

SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.max.dynamic.partitions=1000;
SET hive.exec.max.dynamic.partitions.pernode=1000;

DROP TABLE IF EXISTS Logs;

CREATE EXTERNAL TABLE Logs (
    ip           STRING,
    http_request STRING,
    page_size    SMALLINT,
    status_code  SMALLINT,
    information  STRING
)
PARTITIONED BY (request_time STRING)
STORED AS TEXTFILE;

INSERT INTO TABLE Logs PARTITION (request_time)
SELECT ip, http_request, page_size, status_code, information, request_time FROM LogsTmp;

SELECT * FROM Logs LIMIT 10;

DROP TABLE IF EXISTS IPRegions;

CREATE EXTERNAL TABLE IPRegions (
    ip     STRING,
    region STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
WITH SERDEPROPERTIES (
    "input.regex" = '^(\\S*)\\t(\\S*).*$'
)
STORED AS TEXTFILE
LOCATION '/data/user_logs/ip_data_M';

SELECT * FROM IPRegions LIMIT 10;

DROP TABLE IF EXISTS Users;

CREATE EXTERNAL TABLE Users (
    ip      STRING,
    browser STRING,
    gender  STRING,
    age     TINYINT
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
WITH SERDEPROPERTIES (
    "input.regex" = '^(\\S*)\\t(\\S*)\\t(\\S*)\\t(\\d*).*$'
)
STORED AS TEXTFILE
LOCATION '/data/user_logs/user_data_M';

SELECT * FROM Users LIMIT 10;

DROP TABLE IF EXISTS Subnets;

CREATE EXTERNAL TABLE Subnets (
    ip   STRING,
    mask STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
WITH SERDEPROPERTIES (
    "input.regex" = '^(\\S*)\\t(\\S*).*$'
)
STORED AS TEXTFILE
LOCATION '/data/subnets/variant1';

SELECT * FROM Subnets LIMIT 10;

