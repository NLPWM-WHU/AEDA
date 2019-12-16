from math import pow

from AISD_addALL_TrainTest_Pretrain_df.config import neg_table_size


def InitNegTable(edges):
	text, user, item, user_loc, item_loc, review_rating, item_rating,user_jdate,review_date,item_price,item_wifi,item_acc,item_web,item_phone = zip(*edges)
	text_list=list(text)
	user_list=list(user)
	item_list=list(item)
	user_loc_list=list(user_loc)
	item_loc_list=list(item_loc)
	review_rating_list=list(review_rating)
	item_rating_list=list(item_rating)
	user_jdate_list = list(user_jdate)
	review_date_list = list(review_date)
	item_price_list = list(item_price)
	item_wifi_list = list(item_wifi)
	item_acc_list = list(item_acc)
	item_web_list = list(item_web)
	item_phone_list = list(item_phone)

	loc_list = user_loc_list
	loc_list.extend(item_loc_list)

	rating_list = review_rating_list
	rating_list.extend(item_rating_list)

	date_list = user_jdate_list
	date_list.extend(review_date_list)


	text_neg_table,num_texts = getNegTable(text_list)
	user_neg_table,num_users = getNegTable(user_list)
	item_neg_table,num_items = getNegTable(item_list)
	loc_neg_table,num_locs = getNegTable(loc_list)
	rating_neg_table,num_ratings = getNegTable(rating_list)

	date_neg_table,num_dates = getNegTable(date_list)
	price_neg_table,num_prices = getNegTable(item_price_list)
	wifi_neg_table,num_wifis = getNegTable(item_wifi_list)
	acc_neg_table,num_accs = getNegTable(item_acc_list)
	web_neg_table,num_webs = getNegTable(item_web_list)
	phone_neg_table,num_phones = getNegTable(item_phone_list)


	return text_neg_table,num_texts,user_neg_table,num_users,item_neg_table,num_items\
		,loc_neg_table,num_locs,rating_neg_table,num_ratings,date_neg_table,num_dates\
		,price_neg_table,num_prices,wifi_neg_table,num_wifis,acc_neg_table,num_accs\
		,web_neg_table,num_webs,phone_neg_table,num_phones


def getNegTable(node):
	NEG_SAMPLE_POWER = 0.75
	node_degree = {}
	for i in node:
		if i in node_degree:
			node_degree[i] += 1
		else:
			node_degree[i] = 1
	sum_degree = 0
	for i in node_degree.values():
		sum_degree += pow(i, 0.75)

	por = 0
	cur_sum = 0
	vid = -1
	neg_table = []
	degree_list = list(node_degree.values())
	node_id = list(node_degree.keys())
	for i in range(neg_table_size):
		if (((i + 1) / float(neg_table_size)) > por):
			cur_sum += pow(degree_list[vid + 1], NEG_SAMPLE_POWER)
			por = cur_sum / sum_degree
			vid += 1
		neg_table.append(node_id[vid])
	print(len(neg_table),len(node_degree))
	return neg_table,len(node_degree)
