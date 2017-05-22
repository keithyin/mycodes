create database basket 
character set 'utf8'
collate 'utf8_unicode_ci';
	;
use basket;
#创建user表
create table basket_user(
	user_id int unsigned not null auto_increment primary key,
	user_password varchar(20)  not null,
	user_tele varchar(20)  not null,
	user_level int unsigned not null,    # 1,root  2,admin  3,common
	user_email varchar(50) not null,
	user_nickname varchar(50) not null,
	head_portrait varchar(50) not null
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table group_name(
	id int unsigned not null auto_increment primary key,
	name varchar(10) default '**'
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table course_character(
	id int unsigned not null auto_increment primary key,
	name varchar(10) not null default '**'
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table personal_information(
	id int unsigned not null auto_increment primary key,
	u_id int unsigned not null,
	constraint foreign key(u_id) references basket_user(user_id),
	real_name varchar(20) not null,
	sex tinyint not null, #0:man 1:woman
	birthday date not null,
	idcard varchar(18) not null,
	nation varchar(10) not null,
	wordplace varchar(10) not null,
	department varchar(10) not null,
	rank varchar(10) not null,
	highest_degree varchar(10) not null,
	email varchar(30) not null,
	telephone_number varchar(11) not null,
	group_id int unsigned not null,
	constraint foreign key(group_id) references group_name(id),
	user_total_score int default 0,
	user_total_workload real default 0.0
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;
#多人合作项目 分值分配标准表
create table distribution_standard(id int unsigned not null auto_increment primary key,
	one float(3,2) not null,
	two float(3,2) ,
	three float(3,2),
	four float(3,2),
	five float(3,2),
	six float(3,2),
	seven float(3,2)
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;
# 计分标准表

create table scoring_formula(id int unsigned not null auto_increment primary key,
	type int,                   #1,理工类  2,社科类 3,艺术作品类 4,教学研究工作
	project varchar(50) not null,
	content varchar(50)  not null,
	basic_score int unsigned not null,
	describing varchar(100) not null
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;
#user 成就表

create table achievement(
	id int unsigned not null auto_increment primary key,
	u_id int unsigned not null,
	constraint foreign key(u_id) references basket_user(user_id),
	cf_id int unsigned not null,
	constraint foreign key(cf_id) references scoring_formula(id),
	syn_position tinyint not null,  # 1-7 表示各顺位
	project_sources varchar(50) not null,
	serial_number varchar(50) default '***',
	project_name varchar(20) not null,
	date_begin varchar(10) not null,
	date_end varchar(10) not null,
	fund real default 0,
	pro_desc varchar(100) default '0',
	f real default 0.0,
	status int not null default 0,   # 0：在研  1：結題
	score int unsigned not null ,# 成就提供的分值
	checked tinyint default 0,  #用户提交的成就是否被验证 0 未验证，1 验证失败，2 验证通过
	teammate varchar(50) not null
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table influence(id int unsigned not null auto_increment primary key,
	name varchar(20) not null,
	value int not null
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table education_experience(id int unsigned not null auto_increment primary key,
	uer_id int unsigned not null,
	date_begin varchar(10) not null,
	date_end varchar(10) not null default 0,
	university varchar(20) not null,
	department varchar(30) not null,
	degree varchar(10) not null,
	teacher varchar(10) not null
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table work_experience(id int unsigned not null auto_increment primary key,
	uer_id int unsigned not null,
	date_begin varchar(10) not null,
	date_end varchar(10) default '0000/00',
	company varchar(20) not null,
	department varchar(20) not null,
	position varchar(20) not null
)engine=InnoDb default charset=utf8 collate=utf8_unicode_ci;



create table curriculum(
	id int unsigned not null auto_increment primary key,
	serial_number varchar(20) default '',
	name varchar(20) not null
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table teach(
	id int unsigned not null auto_increment primary key,
	type int not null default 1, # 1=>k1 .......
	curriculum_id int unsigned not null,
	constraint foreign key(curriculum_id) references curriculum(id),
	u_id int unsigned not null,
	constraint foreign key(u_id) references basket_user(user_id),
	date_begin_end varchar(20) not null default '****',
	department varchar(10) default '',
	course_character varchar(2) default '必修',
	credit int default 0,
	total_class_hours int default 0,
	theory_class_hours int default 0,
	practice_class_hours int default 0,
	com_class_hours int default 0,
	number_of_students int default 0,
	number_of_classes int default 0,
	classes varchar(50) default '**'Te
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table teach_factor(
	id int unsigned  not null auto_increment primary key,
	value_1 real default 0.0,
	value_2 real default 0.0,
	value_3 real default 0.0
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table workload_factor_distribution(
	id int unsigned not null auto_increment primary key,
	t_id int unsigned not null,
	constraint foreign key(t_id) references teach(id),
	k1 real default 0,
	k2 real default 0,
	k3 real default 0,
	k4 real default 0,
	k5 real default 0,
	k6 real default 0,
	k7 real default 0
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table workload_distribution(
	id int unsigned not null auto_increment primary key,
	t_id int unsigned not null,
	constraint foreign key(t_id) references teach(id),
	workload_of_theory real default 0.0,
	workload_of_practice real default 0.0,
	workload_of_cum real default 0.0,
	workload_of_total real default 0.0
)engine=InnoDB default charset=utf8 collate=utf8_unicode_ci;

create table achi_pic(
	id int unsigned not null auto_increment primary key,
	achi_id int unsigned not null,
	constraint foreign key(achi_id) references achievement(id),
	path varchar(20) not null
)engine=InnoDb default charset=utf8 collate=utf8_unicode_ci;




# user表用来存放用户基本信息, 用户名,密码,手机号,总分, 级别（普通用户,root, 管理员）
# distribution_standard 用来存放,多人合作分值分配的百分比
# scoring_formula存放各个项目的分值和相关描述
# achiement 存放用户的成就,每一行只存放一个成就