# chinese_slot_filling
模型训练 python model/model_train.py {domain}
模型测试 python model/model_test.py {domain}
domain目前只支持music

方法参考Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling，输入是分词结果并加入知识库知识, intent detection部分加入global attention，slot filling部分拼接知识库信息
