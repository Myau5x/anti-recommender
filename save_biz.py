from src.pipe_spar import api_f, cluster_biz_by_review , cluster_user_by_review


b0 = rests_id.join(biz_with_cl.filter('biz_cl ==0'), 'business_id', how = 'left')
b0 = b0.withColumnRenamed('biz_cl', 'cl_0')
for i in range(1,18):
    cond = 'biz_cl =='+str(i)
    colName = 'cl_'+str(i)
    b0 = b0.join(biz_with_cl.filter(cond), 'business_id', how = 'left').withColumnRenamed('biz_cl', colName)
biz_df_cl = rests.select(api_f).join(b0, 'business_id').toPandas()
biz_df_cl.to_csv('biz_cluster.csv')
