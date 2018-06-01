 
 
--4号线的乘客最爱去哪里消费
select *
from (
select  mchnt_cd,card_accptr_nm_addr,count(*) as trans_num
from sh_railway_mchnt_test_v4
where ls_mchnt_cd='102310041110011'
group by mchnt_cd,card_accptr_nm_addr
)a order by trans_num desc limit 1000;
 
--哪些商户的乘客最爱乘4号线
 
select *
from (
select  ls_mchnt_cd,ls_card_accptr_nm_addr,count(*) as trans_num
from sh_railway_mchnt_test_v4
where mchnt_cd='102310041110011'
group by ls_mchnt_cd,ls_card_accptr_nm_addr
)a order by trans_num desc limit 1000;
 
 
 
Sql供参考：
use  00012900_shanghai;
drop table sh_railway_mchnt_test_v1;
create table sh_railway_mchnt_test_v1 as
SELECT pri_acct_no_conv,
       mchnt_tp,
       mchnt_cd,
       card_accptr_nm_addr,
       concat('2018',tfr_dt_tm) AS tfr_dt_tm,
       to_ts,
       lag(concat('2018',tfr_dt_tm),1,"") over (partition BY pri_acct_no_conv
                                                ORDER BY tfr_dt_tm) AS ls_tfr_dt_tm,
       lag(mchnt_tp,1,"") over (partition BY pri_acct_no_conv
                                ORDER BY tfr_dt_tm) AS ls_mchnt_tp,
       lag(mchnt_cd,1,"") over (partition BY pri_acct_no_conv
                                ORDER BY tfr_dt_tm) AS ls_mchnt_cd,
       lag(card_accptr_nm_addr,1,"") over (partition BY pri_acct_no_conv
                                           ORDER BY tfr_dt_tm) AS ls_card_accptr_nm_addr
FROM `00012900_shanghai`.tbl_common_his_trans_success_parquet
WHERE pdate>='20180101';
 
describe sh_railway_mchnt_test_v1;
 
drop table sh_railway_mchnt_test_v2;
create table sh_railway_mchnt_test_v2 as
 
SELECT
      mchnt_tp,
      mchnt_cd,
      card_accptr_nm_addr,
       tfr_dt_tm,
       to_ts,
       ls_tfr_dt_tm,
      ls_mchnt_tp,
      ls_mchnt_cd,
      ls_card_accptr_nm_addr,
      round(( unix_timestamp(tfr_dt_tm,'yyyyMMddHHmmss')-unix_timestamp(ls_tfr_dt_tm,'yyyyMMddHHmmss'))/60) as time_diff
FROM sh_railway_mchnt_test_v1
WHERE ls_tfr_dt_tm!=''
  AND mchnt_cd!=ls_mchnt_cd
 
  
drop table   sh_railway_mchnt_test_v3;
CREATE TABLE sh_railway_mchnt_test_v3 AS
SELECT *
FROM sh_railway_mchnt_test_v2
WHERE (mchnt_cd IN ('102310041110010',
                    '102310041110011',
                    '102310041110012',
                    '104310041119134',
                    '104310041119135',
                    '104310041119136',
                    '105290000025237',
                    '105290000025238',
                    '105290000025239',
                    '105290000025240',
                    '113310041110003',
                    '113310041110004',
                    '113310041110005',
                    '113310041110006',
                    '301310041310009',
                    '301310041310010',
                    '301310041310011',
                    '301310041310011')
       OR ls_mchnt_cd IN ('102310041110010',
                          '102310041110011',
                          '102310041110012',
                          '104310041119134',
                          '104310041119135',
                          '104310041119136',
                          '105290000025237',
                          '105290000025238',
                          '105290000025239',
                          '105290000025240',
                          '113310041110003',
                          '113310041110004',
                          '113310041110005',
                          '113310041110006',
                          '301310041310009',
                          '301310041310010',
                          '301310041310011',
                          '301310041310011'))
  AND NOT ((mchnt_cd IN ('102310041110010',
                         '102310041110011',
                         '102310041110012',
                         '104310041119134',
                         '104310041119135',
                         '104310041119136',
                         '105290000025237',
                         '105290000025238',
                         '105290000025239',
                         '105290000025240',
                         '113310041110003',
                         '113310041110004',
                         '113310041110005',
                         '113310041110006',
                         '301310041310009',
                         '301310041310010',
                         '301310041310011',
                         '301310041310011')
            AND ls_mchnt_cd IN ('102310041110010',
                                '102310041110011',
                                '102310041110012',
                                '104310041119134',
                                '104310041119135',
                                '104310041119136',
                                '105290000025237',
                                '105290000025238',
                                '105290000025239',
                                '105290000025240',
                                '113310041110003',
                                '113310041110004',
                                '113310041110005',
                                '113310041110006',
                                '301310041310009',
                                '301310041310010',
                                '301310041310011',
                                '301310041310011')));
 
drop table sh_railway_mchnt_test_v4;
CREATE TABLE sh_railway_mchnt_test_v4 AS
SELECT *
FROM sh_railway_mchnt_test_v3
where time_diff<=120;
 
 
select * from sh_railway_mchnt_test_v4 limit 100














drop table xrli_diff;
create table xrli_diff as
SELECT
	  pri_acct_no_conv,
      mchnt_tp,
      mchnt_cd,
      card_accptr_nm_addr,
       tfr_dt_tm,
       to_ts,
       ls_tfr_dt_tm,
      ls_mchnt_tp,
      ls_mchnt_cd,
      ls_card_accptr_nm_addr,
      round(( unix_timestamp(tfr_dt_tm,'yyyyMMddHHmmss')-unix_timestamp(ls_tfr_dt_tm,'yyyyMMddHHmmss'))/60) as time_diff
FROM sh_railway_mchnt_test_v1
WHERE ls_tfr_dt_tm!=''
  AND mchnt_cd!=ls_mchnt_cd
  
  
  
  
  
收单机构	商户编号	地铁线路
建行	105290000025237	1号线
农行	113310041110003	2号线
工行	102310041110010	3号线
工行	102310041110011	4号线
建行	105290000025238	5号线
中行	104310041119134	6号线
工行	102310041110012	7号线
交行	301310041310010	8号线
中行	104310041119135	8号线
建行	105290000025239	9号线
建行	105290000025240	10号线
农行	113310041110004	11号线
中行	104310041119136	12号线
农行	113310041110005	13号线
交行	301310041310009	16号线
农行	113310041110006	17号线
交行	301310041310011	磁浮线