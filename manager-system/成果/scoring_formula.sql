use basket;
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '纵向科研项目', '国家级:重大', 3500, '国家935项目主课题,863重大项目,自然科学基金
	重大项目,国家杰出青年青年基金项目,国家科技支撑重大项目,国家科技部重大项目等。'
);

insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '纵向科研项目', '国家级:重点', 2500, '国家973项目子课题,863项目,自然科学基金重点项目
	国家科技支撑计划项目,国家科技部重点项目等。'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '纵向科研项目', '国家级:一般', 1500, '国家自然科学基金项目,国家社科基金项目,
	国家科技部一般项目等。'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '纵向科研项目', '省部级', 700, '杰出青年基金项目,省科技攻关计划项目,省自然科学基金（博士
	基金）项目,教育部项目,教育部人才基金项目,霍英东青年教师基金项目,中国博士后基金,国务院
	个部委科研计划项目等。'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '纵向科研项目', '市厅级', 400, '省博士后基金,教育部出国留学人员基金,青岛市科技局科技
	发展计划项目,各地市（地级市）科技局科技计划项目等。'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '纵向科研项目', '其它项目', 200, '国家重点实验室开放基金,其它纵向项目等（不含校级项目'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '横向项目' , '到校经费', 30, '30/万元, 指当年度实际到学校指定账号的横向项目经费额。'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '国家级:一等奖:国家自然科学奖', 22000, '国家自然科学奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '国家级:一等奖:国家发明奖', 20000, '国家发明奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '国家级:一等奖:国家科技进步奖', 18000, '国家科技进步奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '国家级:二等奖:国家自然科学奖', 14000, '国家自然科学奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '国家级:二等奖:国家发明奖', 12000, '国家发明奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '国家级:二等奖:国家科技进步奖', 10000, '国家科技进步奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '省级部:一等奖', 6000, '山东省自然科学奖,山东省科技进步奖, 山东省技术
	发明奖,中国高校优秀成果奖,中国石油化工工业协会科技奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '省级部:二等奖', 3000, '山东省自然科学奖,山东省科技进步奖, 山东省技术
	发明奖,中国高校优秀成果奖,中国石油化工工业协会科技奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '省级部:三等奖', 2000, '山东省自然科学奖,山东省科技进步奖, 山东省技术
	发明奖,中国高校优秀成果奖,中国石油化工工业协会科技奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '市厅级:一等奖', 1400, '青岛市自然科学奖,青岛市发明奖,青岛市科技进步奖,
	省高校优秀成果奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '市厅级:二等奖', 900, '青岛市自然科学奖,青岛市发明奖,青岛市科技进步奖,
	省高校优秀成果奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '市厅级:三等奖', 600, '青岛市自然科学奖,青岛市发明奖,青岛市科技进步奖,
	省高校优秀成果奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '其它:一等奖', 60, '其它市厅级及学校认定的协会（学会）,科研成果奖励等
	（不包含校级奖励,校级奖励按其它文件执行）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '其它:二等奖', 40, '其它市厅级及学校认定的协会（学会）,科研成果奖励等
	（不包含校级奖励,校级奖励按其它文件执行）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '科研成果获奖', '其它:三等奖', 20, '其它市厅级及学校认定的协会（学会）,科研成果奖励等
	（不包含校级奖励,校级奖励按其它文件执行）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '项目鉴定', '省部级鉴定', 200, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '项目鉴定', '市厅级鉴定', 100, ' '
	);
	insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '获奖:国际金奖', 600, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '获奖:国际银奖', 400, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '获奖:国际铜奖', 200, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '获奖:国内金奖', 400, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '获奖:国内银奖', 200, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '获奖:国内铜奖', 80, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '发明:国内', 800, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '发明:国外', 600, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '其它专利:国外', 300, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '其它专利:国内:实用新型', 200, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '专利', '其它专利:国内:外观新型', 100, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '学术论文', '公开发表:SCIENCE_NATURE', 10000, 'SCIENCE,NATURE'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '学术论文', '公开发表:SCI检索系统收录', 200, ' SCI检索系统收录 200+F*30  F为影响因子'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '学术论文','公开发表:EI_ISTP等检索系统收录', 200, 'EI,ISTP等检索系统收录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '学术论文','公开发表:A类', 150, '青岛科技大学规定A类期刊目录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '学术论文','公开发表:B类', 100, '青岛科技大学规定B类期刊目录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '学术论文','公开发表:C类', 50, '青岛科技大学规定C类期刊目录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '著作', '专著:外文版', 1200, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '著作', '专著:中文版', 900, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '著作', '译者编译', 300, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '著作', '编著', 600, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '著作', '教材', 400, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	1, '著作', '科普', 300, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '纵向科研项目', '国家级:重大', 3500, '国家社会科学基金重大项目,国家自然科学基金
	重大项目,国家杰出青年青年基金项目,国家科技支撑重大项目,国家科技部重大项目等。'
);

insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '纵向科研项目', '国家级:重点', 2500, '国家社会科学基金重点项目,国家自然科学基金重点项目
	国家科技支撑计划项目,国家科技部重点项目等。'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '纵向科研项目', '国家级:一般', 1500, '国家自然科学基金项目,国家社科基金项目,
	国家社科基金艺术学项目,国家软科学项目等。'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '纵向科研项目', '省部级', 700, '省杰出青年基金项目,省科技攻关计划项目,省自然科学基金（博士
	基金）项目,省社科规划项目,全国教育科学规划项目,教育部人文社科规划项目,教育部人才基金项目,霍英东青年教师基金项目,中国博士后基金,国务院
	个部委科研计划项目,中国博士后基金等。'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '纵向科研项目', '市厅级', 400, '青岛市社科规划项目,青岛市双百调研项目,教育部出国留学人员基金,
	青岛市科技局科技发展计划（含软科学）项目,山东省教育厅人文社科计划项目,山东省教育厅科技计划
	项目,山东省教育科学项目,各地市（地级市）科技局科技计划（软科学）项目,省博士后基金等。'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '纵向科研项目', '其它项目', 200, '其它纵向项目等（不含校级项目）'
);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '横向项目' , '到校经费', 30, '30/万元, 指当年度实际到学校指定账号的横向项目经费额。'
	);
/*
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '国家级:一等奖', 22000, '国家自然科学奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '国家级:一等奖', 20000, '国家发明奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '国家级:一等奖', 18000, '国家科技进步奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '国家级:二等奖', 14000, '国家自然科学奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '国家级:二等奖', 12000, '国家发明奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '国家级:二等奖', 10000, '国家科技进步奖'
	);
*/
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '省级部:一等奖', 6000, '山东省科技进步（软科学）奖,山东省社科优秀成果奖,
	全国青年社科优秀成果奖,中国高校优秀成果奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '省级部:二等奖', 3000, '山东省科技进步（软科学）奖,山东省社科优秀成果奖,
	全国青年社科优秀成果奖,中国高校优秀成果奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '省级部:三等奖', 2000, '山东省科技进步（软科学）奖,山东省社科优秀成果奖,
	全国青年社科优秀成果奖,中国高校优秀成果奖等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '市厅级:一等奖', 1400, '青岛市科技进步（含软科学）奖,青岛市社科优秀成果奖
	山东省软科学优秀成果奖,省高校优秀成果奖,国家部委优秀科技成果奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '市厅级:二等奖', 900, '青岛市科技进步（含软科学）奖,青岛市社科优秀成果奖
	山东省软科学优秀成果奖,省高校优秀成果奖,国家部委优秀科技成果奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '市厅级:三等奖', 600, '青岛市科技进步（含软科学）奖,青岛市社科优秀成果奖
	山东省软科学优秀成果奖,省高校优秀成果奖,国家部委优秀科技成果奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '其它:一等奖', 60, '其它市厅级及学校认定的协会（学会）,科研成果奖励等
	（不包含校级奖励,校级奖励按其它文件执行）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '其它:二等奖', 40, '其它市厅级及学校认定的协会（学会）,科研成果奖励等
	（不包含校级奖励,校级奖励按其它文件执行）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '科研成果获奖', '其它:三等奖', 20, '其它市厅级及学校认定的协会（学会）,科研成果奖励等
	（不包含校级奖励,校级奖励按其它文件执行）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '项目鉴定', '省部级鉴定', 200, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '项目鉴定', '市厅级鉴定', 100, ' '
	);
/*
	insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '获奖:国际金奖', 600, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '获奖:国际银奖', 400, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '获奖:国际铜奖', 200, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '获奖:国内金奖', 400, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '获奖:国内银奖', 200, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '获奖:国内铜奖', 80, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '发明:国内', 800, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '发明:国外', 600, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '其它专利:国外', 300, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '其它专利:国内:实用新型', 200, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '专利', '其它专利:国内:外观新型', 100, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
*/
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '学术论文', '公开发表:中国社会科学', 1000, '中国社会科学'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '学术论文', '公开发表:SSCI检索系统收录', 200, ' SSCI检索系统收录 200+F*30  F为影响因子'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '学术论文', '公开发表', 300, '新华文摘全文转载,中国人民大学
	报刊复印资料全文转载'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '学术论文','公开发表', 200, 'CSSCI收录,EI收录,ISTP（ISSHI）收录,人民日报,光明日报
	经济日报理论板（4000+字）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '学术论文','公开发表:A类', 150, '青岛科技大学规定A类期刊目录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '学术论文','公开发表:B类', 100, '青岛科技大学规定B类期刊目录,其它省级及以上党报理论板
	（3000+字）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '学术论文','公开发表:C类', 50, '青岛科技大学规定C类期刊目录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '著作', '专著:外文版', 1200, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '著作', '专著:中文版', 900, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '著作', '译者编译', 300, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '著作', '编著', 600, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '著作', '教材', 400, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	2, '著作', '科普_文学作品', 300, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '纵向科研项目', '国家级', 1500, '国家社会科学基金（艺术类）项目,国家社科基金艺术项目
	,国家软科学项目'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '纵向科研项目', '省部级', 700, '山东省社科规划（艺术类）项目,省博士基金, 教育部人文
	社科规划（艺术类）项目,中国博士后基金等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '纵向科研项目', '市厅级', 400, '青岛市社科规划（艺术类）项目,青岛市双百调研项目,山东
	省文化厅科研项目,山东省教育厅人文社科项目,山东省教育科学项目,各地市（地级市）科技局科技计划
	（软科学）项目,省博士后基金等'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '纵向科研项目', '其它项目', 200, '其它纵向项目'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '横向经费', ' ', 30, '(30+40)/万元, 指当年度实际到学校指定账号的横向项目经费额。'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '专利', '国外', 300, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '专利', '国内:实用新型', 200, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '专利', '国外:外观设计', 100, '获得中华人民共和国专利的职务发明（以有专利公示号为准）
	且我校具有自主知识产权（职务发明）的专利'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '论文', ' ', 300, '新华文摘（全文转载）,中国人民大学报刊复印材料（全文转载）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '论文', 'SSCI_AHCI系统收录', 200, '(200+IF*30)SSCI,AHCI系统收录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '论文', 'CSSCI_ISSHI_EI', 200, 'CSSCI收录,ISSHI,EI收录论文被国家级文化,艺术专业博物馆
	,美术馆收藏的作品'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '论文', 'A类', 150, '青岛科技大学规定A类期刊目录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '论文', 'B类', 100, '青岛科技大学规定B类期刊目录（含中国美术家,音乐学会各艺委员会杂志）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '论文', 'C类', 50, '青岛科技大学规定A类期刊目录'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '其它', ' ', 300, '国家级个人艺术展'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '其它', ' ', 200, '国外个人艺术邀请展'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '其它', ' ', 150, '省级个人艺术展'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '其它', ' ', 100, '市厅级个人艺术展,（含青岛科技大学美术馆,校级仅用于考核,
	不参加分配）'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:金奖', 12000, '全国美术展览,全国音乐比赛金奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:银奖', 10000, '全国美术展览,全国音乐比赛银奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:铜奖', 6000, '全国美术展览,全国音乐比赛铜奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:优秀奖', 3000, '全国美术展览,全国音乐比赛优秀奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:一等奖', 1400, '全国美术展览,全国音乐比赛入选奖,国家级单项一等奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:二等奖', 900, '国家级单项二等奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:三等奖', 600, '国家级单项三等奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:优秀奖', 400, '国家级单项优秀奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '国家级:入选奖', 200, '国家级单项入选奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:一等奖', 6000, '山东省社科优秀成果奖,全国青年社科优秀成果奖,
	中国高校优秀成果奖,山东省泰山文艺奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:二等奖', 3000, '山东省社科优秀成果奖,全国青年社科优秀成果奖,
	中国高校优秀成果奖,山东省泰山文艺奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:三等奖', 2000, '山东省社科优秀成果奖,全国青年社科优秀成果奖,
	中国高校优秀成果奖,山东省泰山文艺奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:美协音协一等奖', 1400, '含同级别艺术专业展演,山东省文化厅、山东省
	文化艺术科学优秀成果奖,山东省软科学优秀成果奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:美协音协二等奖', 900, '含同级别艺术专业展演,山东省文化厅、山东省
	文化艺术科学优秀成果奖,山东省软科学优秀成果奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:美协音协三等奖', 600, '含同级别艺术专业展演,山东省文化厅、山东省
	文化艺术科学优秀成果奖,山东省软科学优秀成果奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:美协音协入选奖', 400, '含同级别艺术专业展演,山东省文化厅、山东省
	文化艺术科学优秀成果奖,山东省软科学优秀成果奖'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:单项奖一等奖', 600, '含山东省教育厅举办的音乐、美术基本功大赛'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:单项奖二等奖', 400, '含山东省教育厅举办的音乐、美术基本功大赛'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:单项奖三等奖', 200, '含山东省教育厅举办的音乐、美术基本功大赛'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:单项奖优秀奖', 150, '含山东省教育厅举办的音乐、美术基本功大赛'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '省部级:单项奖入选奖', 100, '含山东省教育厅举办的音乐、美术基本功大赛'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '其它级别:一等奖', 50, '不含校级奖励,校级奖励按其它文件执行'
	);

insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '其它级别:二等奖', 30, '不含校级奖励,校级奖励按其它文件执行'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '其它级别:三等奖', 20, '不含校级奖励,校级奖励按其它文件执行'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '作品获奖', '其它级别:优秀奖', 10, '不含校级奖励,校级奖励按其它文件执行'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '著作', '专著:外文版', 1200, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定.6.个人书画（作品）集视出版社级别加
	150-200分'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '著作', '专著:中文版', 900, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定.6.个人书画（作品）集视出版社级别加
	150-200分'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '教材', ' ', 400, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定.6.个人书画（作品）集视出版社级别加
	150-200分'
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '个人作品集', ' ', 200, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定.6.个人书画（作品）集视出版社级别加
	150-200分'
	
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	3, '译注_编译', ' ', 300, '1.著作以第一版计分（著作如享受过学校资助者不计算得分）
	2.著作类计分严格按照本人承担章节或字数比例计算（可平均计分）3.著作属操作手册,练习册累的按
	编著教材计分。4.以论文集形式的著作不计算著作分按论文类计算,教材若列为国家级规划教材（以
	批文为准）加200分。5.不同版次,多卷册的著作按ISBN号确定.6.个人书画（作品）集视出版社级别加
	150-200分'
	
	);

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------


insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学成果奖', '国家级:特等奖', 22000, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学成果奖', '国家级:一等奖', 18000, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学成果奖', '国家级:二等奖', 10000, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学成果奖', '省级:一等奖', 6000, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学成果奖', '省级:二等奖', 3000, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学成果奖', '省级:三等奖', 2000, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学成果奖', '校级:一等奖', 700, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学成果奖', '校级:二等奖', 300, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '精品课程', '国家级', 1500, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '精品课程', '省级', 700, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '精品课程', '校级', 300, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '实验教学示范中心', '国家级', 1500, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '实验教学示范中心', '省级', 700, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '实验教学示范中心', '校级', 100, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '品牌与特色专业', '国家级', 1500, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '品牌与特色专业', '省级', 700, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '品牌与特色专业', '校级', 100, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学团队', '国家级', 1500, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学团队', '省级', 700, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学团队', '校', 100, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '人才培养模式创新试验区', '国家级', 1500, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '人才培养模式创新试验区', '省级', 700, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '人才培养模式创新试验区', '校级', 100, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '双语课程', '国家级', 1500, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '双语课程', '省级', 700, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '双语课程', '校级', 100, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学名师奖', '国家级', 10000, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学名师奖', '省级', 1200, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学名师奖', '校级', 300, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学研究立项_教材立项', '国家级:重点', 1500, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学研究立项_教材立项', '国家级:一般', 1200, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学研究立项_教材立项', '国家级:自筹', 900, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学研究立项_教材立项', '省级:重点', 900, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学研究立项_教材立项', '省级:一般', 700, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学研究立项_教材立项', '省级:自筹', 500, ' '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀教材', '国家级', 1500, '与图书馆审核的相同项目不重复计分 '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀教材', '省级:一等奖', 900, '与图书馆审核的相同项目不重复计分 '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀教材', '省级:二等奖', 600, '与图书馆审核的相同项目不重复计分 '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀教材', '省级:三等奖', 400, '与图书馆审核的相同项目不重复计分 '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀教材', '校级:一等奖', 200, '与图书馆审核的相同项目不重复计分 '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀教材', '校级:二等奖', 100, '与图书馆审核的相同项目不重复计分 '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '实验教学改革奖_实验技术成果奖', '省级:一等奖', 900, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '实验教学改革奖_实验技术成果奖', '省级:二等奖', 600, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '实验教学改革奖_实验技术成果奖', '省级:三等奖', 400, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '实验教学改革奖_实验技术成果奖', '校级:一等奖', 200, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '实验教学改革奖_实验技术成果奖', '校级:二等奖', 100, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '多媒体教学课件大赛', '国家级:一等奖', 900, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '多媒体教学课件大赛', '国家级:二等奖', 600, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '多媒体教学课件大赛', '国家级:三等奖', 400, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '多媒体教学课件大赛', '省级:一等奖', 600, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '多媒体教学课件大赛', '省级:二等奖', 400, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '多媒体教学课件大赛', '省级:三等奖', 200, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '多媒体教学课件大赛', '校级:一等奖', 200, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '多媒体教学课件大赛', '校级:二等奖', 100, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '各类学科竞赛等获奖指导老师', '国家级:一等奖', 500, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '各类学科竞赛等获奖指导老师', '国家级:二等奖', 300, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '各类学科竞赛等获奖指导老师', '国家级:三等奖', 100, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '各类学科竞赛等获奖指导老师', '省级:一等奖', 300, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '各类学科竞赛等获奖指导老师', '省级:二等奖', 100, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '各类学科竞赛等获奖指导老师', '省级:三等奖', 60, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀学位论文指导', '国家级:博士', 2000, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀学位论文指导', '省级:博士', 500, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀学位论文指导', '省级:硕士', 300, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀学位论文指导', '省级:学士', 100, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '优秀学位论文指导', '校级', 60, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学效果优秀奖', '校级:一等奖', 500, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学效果优秀奖', '校级:二等奖', 300, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学效果优秀奖', '校级:青年奖', 200, '  '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学督导', '校级及院级组长', 100, ' 100/year '
	);
insert into basket.scoring_formula (type, project, content, basic_score, describing) values (
	4, '教学督导', '院级', 50, ' 50/year '
	);