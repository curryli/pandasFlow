CREATE VIEW stat_xibei AS
select * from (
SELECT *
FROM `00017700_xizang`.tbl_common_his_trans_success_parquet

UNION ALL
SELECT *
FROM `00018500_qinghai`.tbl_common_his_trans_success_parquet

UNION ALL
SELECT *
FROM `00018800_xinjiang`.tbl_common_his_trans_success_parquet

UNION ALL
SELECT *
FROM `00018200_gansu`.tbl_common_his_trans_success_parquet

UNION ALL
SELECT *
FROM `00018700_ningxia`.tbl_common_his_trans_success_parquet
)a



18年线下总交易金额
select  sum(cast(trans_at as double))/100, count(trans_at), count(distinct mchnt_cd) from stat_xibei where pdate > '20171231' and trans_chnl not in ('07', '08')
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
411583556876.23999	121195546	944598

18年线上总交易金额
select  sum(cast(trans_at as double))/100, count(trans_at), count(distinct mchnt_cd) from stat_xibei where pdate > '20171231' and trans_chnl in ('07', '08')
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
4330588948.5	2785172	4540

17年线下总交易金额
select sum(cast(trans_at as double))/100, count(trans_at), count(distinct mchnt_cd) from stat_xibei where pdate > '20161231' and pdate < '20180101' and trans_chnl not in ('07', '08')
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
1229606257058.1001	413486751	1485971

17年线上总交易金额
select sum(cast(trans_at as double))/100, count(trans_at), count(distinct mchnt_cd) from stat_xibei where pdate > '20161231' and pdate < '20180101' and trans_chnl in ('07', '08')
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
13005917461.879999	8038709	12589

16年线下总交易金额
select sum(cast(trans_at as double))/100, count(trans_at), count(distinct mchnt_cd) from stat_xibei where pdate > '20151231' and pdate < '20170101' and trans_chnl not in ('07', '08')
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
1051968940659.78	360553996	1150936

16年线上总交易金额
select sum(cast(trans_at as double))/100, count(trans_at), count(distinct mchnt_cd) from stat_xibei where pdate > '20151231' and pdate < '20170101' and trans_chnl in ('07', '08')
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
3037211833.73	3423077	2996
 
 
各商户类型商户数占比 
select count(distinct mchnt_cd) from stat_xibei where pdate > '20171231' 
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
select mchnt_tp, count(distinct mchnt_cd)/947295 as mcnt_ratio from stat_xibei where pdate > '20171231' and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
group by mchnt_tp order by mcnt_ratio desc;

select count(distinct mchnt_cd) from stat_xibei where pdate > '20161231' and pdate < '20180101'
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
select mchnt_tp, count(distinct mchnt_cd)/1495013 as mcnt_ratio from stat_xibei where pdate > '20161231' and pdate < '20180101' and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
group by mchnt_tp order by mcnt_ratio desc;

select count(distinct mchnt_cd) from stat_xibei where pdate > '20151231' and pdate < '20170101'
and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
select mchnt_tp, count(distinct mchnt_cd)/1153873 as mcnt_ratio from stat_xibei where pdate > '20151231' and pdate < '20170101' and trans_id in ("D22","S22","V52","I20","I22","I52","S45","S56","D56","I56","S57","V86","D46","S46","V76","D13","S13","V13","S84","S36","V63","S54","V84","S37","V57","S10","V40","S65","V66","S48","V78","S20","V50","S80","V81","S81","S35","S67","V69","S73","V73","S71","S49","V79","S50","S83","V83","S30","E74","E84","S75","S60","D35","S47","D33","D32","S39","D76")
group by mchnt_tp order by mcnt_ratio desc;
 