<?php
namespace app\controllers;
use yii\web\Controller;
use app\models\Basket_user;
use yii\db\ActiveQuery;
use app\models\Education_experience;
use app\models\Scoring_formula;
use app\models\Distribution_standard;
use app\models\Influence;
use app\models\Personal_information;
use app\models\Achievement;
use yii\helpers\Json;
use app\models\Work_experience;
use app\models\Group;
use app\models\TeachExp;
use app\models\WorkloadFactorDistribution;
use app\models\TeachFactor;
use app\models\WorkloadDistribution;
use app\models\Curriculum;

/**
 * mobile teminal controller
 * @author KeithYin
 *
 */
class MobileController extends Controller{
	/**
	 * 口令
	 * @var unknown
	 */
	private $token='lalala';
		
	/**
	 * login
	 * @param unknown $token 
	 * @param unknown $email 
	 * @param unknown $pwd  
	 * @param unknown $user_level 1:root 2:admin 3:common
	 * @return Json|int  success:return return the json data 0f usre . failed: return 0
	 */
	public function actionPLogin($token,$email,$pwd){  //email   pwd
		if(\Yii::$app->request->isGet){
			$rec['token']=trim($token);
			if($rec['token'] != $this->token)
				return 'illegal token';
			$rec['email'] = $email;  
			$rec['pwd']=$pwd;
//			$rec['user_level']=(int)trim($user_level);
			$exist = Basket_user::find()->where([
					'user_email'=>$rec['email'],
					'user_password'=>$rec['pwd'],
			])->exists();
			if($exist){
				$user = Basket_user::find()->where(['user_email'=>$rec['email']])->asArray()->one();
				return json_encode($user);  //json_encode必须是 array	
			}
			else 
				return 0;
		}else 
			return 'illegal user';
	}
	/**
	 * register
	 * @param unknown $token 
	 * @param unknown $email 
	 * @param unknown $pwd 
	 * @param unknown $tele 
	 * @param unknown $nickname name
	 * @return int|string success:return 1. failed:return reason
	 */
	public function actionPRegister($token,$email,$pwd,$tele,$nickname){
		if(trim($token) != $this->token)
			return 'get out of here';
		$new_user = new Basket_user(['scenario'=>'p_register']);
		if($new_user->addUser($email, $pwd, $tele, $nickname)){
			$user_id = $new_user['user_id'];
			$new_personal_info = new Personal_information();
			return $new_personal_info->addPersonInfo($user_id);
		}else 
			return 'error';
			
	}
	
	/**
	 * modify password
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $old_pwd
	 * @param unknown $new_pwd
	 * @return string|number success:return 1. failed:return reason
	 */
	public function actionPModifyPwd($token,$user_id,$new_pwd,$old_pwd='0'){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$user = Basket_user::findOne((int)trim($user_id));
			//		if($user['user_password']!=trim($old_pwd))
				//			return 'wrong password';
		return $user->modifyPwd($new_pwd);
	}
	
	/**
	 * modify count information
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $nickname
	 * @param unknown $tele
	 * @return string|json|number success:return 1. failed:return reason
	 */
	public function actionPModifyCount($token,$user_id,$nickname,$tele){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$user = Basket_user::findOne((int)trim($user_id));
		return $user->modifyCount($nickname, $tele);
	}
	
	/**
	 * show and update user information
	 * @param unknown $type 1:show 2:update
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $real_name
	 * @param unknown $sex
	 * @param unknown $birthday
	 * @param unknown $idcard
	 * @param unknown $nation
	 * @param unknown $workplace
	 * @param unknown $rank
	 * @param unknown $highest_degree
	 * @param unknown $email
	 * @param unknown $telephone_number
	 * @return int|string   success:return 1. failed:return reason
	 */
	public function actionPUpdatePersonalInformation($token,$type,$user_id,$group_id=1,$real_name='0',$sex='男',$birthday='1111-11-11',$idcard='111111111111111111',$nation='..',$workplace='..',$department='..',$rank='..',$highest_degree='..',$email='..',$telephone_number='11111111111'){
		if(trim($token) != $this->token)
			return 'illegal token';
		if((int)trim($type)==1){  //1 :show  2.update
			$personal_info = Personal_information::find()->where(['u_id'=>(int)trim($user_id)])->asArray()->one();
			if($personal_info['sex']==0)
				$personal_info['sex'] = '男';
			else 
				$personal_info['sex']='女';
			return json_encode($personal_info);
		}
		if(trim($sex)=='男')
			$sex = 0;
		else if(trim($sex=='女'))
			$sex=1;
		else 
			return 'illegal sex format';
		if(\Yii::$app->request->isGet){
			$personal_info = Personal_information::find()->where(['u_id'=>trim($user_id)])->one();
			$personal_info['u_id'] = trim($user_id);
			$personal_info['real_name']=trim($real_name);
			$personal_info['sex'] = trim($sex);
			$personal_info['birthday'] = trim($birthday);
			$personal_info['idcard'] = trim($idcard);
			$personal_info['nation'] = trim($nation);
			$personal_info['wordplace'] = trim($workplace);
			$personal_info['rank'] = trim($rank);
			$personal_info['highest_degree'] = trim($highest_degree);
			$personal_info['email'] = trim($email);
			$personal_info['telephone_number'] =trim($telephone_number);
			$personal_info['department'] = trim($department);
			$personal_info['group_id']=(int)trim($group_id);
			if($personal_info->validate()){
				if($personal_info->save())
					return 1;
			}else
				return json_encode($personal_info->getErrors());		
		}else
			return 'get out of here';
	}
	/**
	 * get id and name of all users
	 * @param unknown $token
	 * @return string success:return json data .failed:return reason
	 */
	public function actionPUsers($token){
		if(trim($token) != $this->token)
			return 'illegal token';
		$res = Basket_user::find()->select(['user_id','user_nickname'])->asArray()->all();
		return json_encode($res);
	}
	
	/**
	 * find user by key word of name,
	 * @param unknown $token
	 * @param unknown $name
	 * @return string|json success: return json data . failed return reason
	 */
	public function actionPFindUserByName($token,$name){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$res = Basket_user::find()->select(['user_id','user_nickname'])->where(['like','user_nickname',$name])->asArray()->all();
		return json_encode($res);
	}
	
	/**
	 * add users' educational information
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $date_begin
	 * @param unknown $date_end
	 * @param unknown $university
	 * @param unknown $department
	 * @param unknown $degree
	 * @param unknown $teacher
	 * @return string|int success:return 1. failed:return reason
	 */
	public function actionPAddEducationExperience($token,$user_id,$date_begin,$date_end,$university,$department,$degree,$teacher){
		if(trim($token) != $this->token)
			return 'illegal token';
		$new_edu_exp = new Education_experience(['scenario'=>'edu_1']);
		$new_edu_exp['uer_id']=trim($user_id);
		$new_edu_exp['date_begin']=trim($date_begin);
		$new_edu_exp['date_end']=trim($date_end);
		$new_edu_exp['university']=trim($university);
		$new_edu_exp['department']=trim($department);
		$new_edu_exp['degree']=trim($degree);
		$new_edu_exp['teacher']=trim($teacher);
		if($new_edu_exp->save()){
			return 1;
		}else 
			return 'validation failed';
	}
	
	
	/**
	 * show user's educational information
	 * @param unknown $token
	 * @param unknown $user_id
	 * @return string|json success:return json data. failed: return reason
	 */
	public function actionPShowEduExp($token,$user_id){
		if(trim($token) != $this->token)
			return 'illegal token';
		$edu_exp = Education_experience::find()->where(['uer_id'=>trim($user_id)])->orderBy(['date_begin'=>SORT_ASC])->asArray()->all();
		return json_encode($edu_exp);
	}
	/**
	 * modify education experiences
	 * @param unknown $token
	 * @param unknown $id
	 * @param unknown $date_begin
	 * @param unknown $date_end
	 * @param unknown $university
	 * @param unknown $department
	 * @param unknown $degree
	 * @param unknown $teacher
	 * @return number|json|string  1:success  other:failed
	 */
	public function actionPModifyEduExp($token,$id,$date_begin,$date_end,$university,$department,$degree,$teacher){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$exists = Education_experience::find()->where(['id'=>(int)trim($id)])->exists();
		if(!$exists)
			return 'not exists';
		$item = Education_experience::findOne((int)trim($id));
		$item['date_begin']=trim($date_begin);
		$item['date_end']=trim($date_end);
		$item['university']=trim($university);
		$item['department']=trim($department);
		$item['degree']=trim($degree);
		$item['teacher']=trim($teacher);
		if($item->save())
			return 1;
		else 
			return json_encode($item->getErrors());
	}
	
	/**
	 * delete education experience by id
	 * @param unknown $token
	 * @param unknown $id
	 * @return number|json|string  1:success
	 */
	public function actionPDeleteEduExp($token,$id){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$exists = Education_experience::find()->where(['id'=>(int)trim($id)])->exists();
		if(!$exists)
			return 'not exists';
		$item = Education_experience::findOne((int)trim($id));
		if($item->delete())
			return 1;
		else 
			return json_encode($item->getErrors());
	}
	
	/**
	 * add work experiences
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $date_begin
	 * @param unknown $date_end
	 * @param unknown $company
	 * @param unknown $department
	 * @param unknown $position
	 * @return number|json|string  1:success
	 */
	public function actionPAddWorkExp($token,$user_id,$date_begin,$date_end,$company,$department,$position){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$item = new Work_experience(['scenario'=>'submit']);
		$item['uer_id']=(int)trim($user_id);
		$item['date_begin']=trim($date_begin);
		$item['date_end']=trim($date_end);
		$item['company']=trim($company);
		$item['department']=trim($department);
		$item['position']=trim($position);
		if($item->save())
			return 1;
		else 
			return json_encode($item->getErrors());
	}
	
	/**
	 * delete work experiences by id
	 * @param unknown $token
	 * @param unknown $id
	 * @return number|json|string  1:success
	 */
	public function actionPDelWorkExp($token,$id){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$exists = Work_experience::find()->where(['id'=>(int)trim($id)])->exists();
		if(!$exists)
			return 'doesn\'s exists';
		$item = Work_experience::findOne((int)trim($id));
		if($item->delete())
			return 1;
		else 
			return json_encode($item->getErrors());
	}
	
	/**
	 * modify work experience
	 * @param unknown $token
	 * @param unknown $id
	 * @param unknown $date_begin
	 * @param unknown $date_end
	 * @param unknown $company
	 * @param unknown $department
	 * @param unknown $position
	 * @return string|number|json  1:sucess
	 */
	public function actionPModifyWorkExp($token,$id,$date_begin,$date_end,$company,$department,$position){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$exists = Work_experience::find()->where(['id'=>(int)trim($id)])->exists();
		if(!$exists)
			return 'donsn\'t exists';
		$item = Work_experience::findOne((int)trim($id));
		$item['date_begin']=trim($date_begin);
		$item['date_end']=trim($date_end);
		$item['company']=trim($company);
		$item['department']=trim($department);
		$item['position']=trim($position);
		if($item->save())
			return 1;
		else 
			return json_encode($item->getErrors());	
	}
	
	/**
	 * show user's work experiences
	 * @param unknown $token
	 * @param unknown $user_id
	 * return json|string
	 */
	public function actionPShowWorkExp($token,$user_id){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$exists = Work_experience::find()->where(['uer_id'=>(int)trim($user_id)])->exists();
		if(!$exists)
			return 'donesn\'t exists';
		$items = Work_experience::find()->where(['uer_id'=>(int)trim($user_id)])->orderBy(['date_begin'=>SORT_DESC])->asArray()->all();
		return json_encode($items);
	}
	
	

	/**
	 * add achievement
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $department
	 * @param unknown $project
	 * @param unknown $content
	 * @param unknown $number
	 * @param unknown $syn_position
	 * @param unknown $project_sources
	 * @param unknown $serial_number
	 * @param unknown $project_name
	 * @param unknown $date_begin
	 * @param unknown $date_end
	 * @param unknown $cal_type   1:科研成果获奖   项目鉴定 专利 . 2:纵向科研项目   3:横向项目  4:学术论文
	 * @param number $fund
	 * @param number $f
	 * @param string $pro_desc
	 * @return string|number success:return 1. failed:return reason
	 */
	//http://localhost/Basket/basic/web/index.php?r=mobile/p-submit=achievement&&token=lalala&&user_id=16&&department=1&&project=%E7%BA%B5%E5%90%91%E7%A7%91%E7%A0%94%E9%A1%B9%E7%9B%AE&&content=%E5%9B%BD%E5%AE%B6%E7%BA%A7:%E9%87%8D%E5%A4%A7&&number=2&&syn_position=1&&project_sources=%E9%9D%92%E5%B9%B4%E5%9F%BA%E9%87%91&&serial_number=BS123456&&project_name=%E4%BC%98%E5%8C%96%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95&&date_begin=2000/09&&date_end=%E8%87%B3%E4%BB%8A&&cal_type=2&&fund=12&&pro_desc=%E5%AD%97%E5%AD%97%E5%AD%97%E5%AD%90
	public function actionPSubmitAchievement($token,$user_id,$department,$project,$content,$number,$syn_position,$project_sources,$serial_number,$project_name,$date_begin,$date_end,$cal_type,$teammate,$fund=0,$f=0,$status=0,$pro_desc='00'){
		if(trim($token) != $this->token)
			return 'illegal token';
		$final_score=0;
		$item = Scoring_formula::find()->where([
				'type'=>trim($department), //1,理工类  2,社科类 3,艺术作品类 4,教学研究工作
				'project'=>trim($project),
				'content'=>trim($content)
		])->one();
		$basic_score = $item['basic_score'];
		$percent_item = Distribution_standard::findOne(trim($number));
		$percent = $percent_item[$this->intToEng(trim($syn_position))];
		$base_multi = Influence::findOne(1)['value'];
		switch((int)$cal_type){
			case 1:  //科研成果获奖   项目鉴定 专利  著作
				$final_score = $basic_score;
				break;
			case 2:  //纵向科研项目
				$final_score = $basic_score+((float)trim($fund)-(float)$basic_score/$base_multi)*$base_multi;
				break;
			case 3: //横向项目
				$final_score = (float)trim($fund)*$basic_score;
				break;
			case 4: //学术论文
				$final_score = $basic_score + (float)$f*$base_multi;
				break;
			default:
				break;
		}
		$final_score*=$percent;
		$final_score = (int) $final_score;
		$new_achi = new Achievement();
		if($new_achi->addAchievement($user_id,$item['id'],$syn_position, $project_sources, $serial_number, $date_begin, $date_end, $project_name, $final_score,$fund,$f,$status,$pro_desc,$teammate))
			return 1;
		else 
			return 'error';
	}
	
	/**
	 * show basic score
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $user_pwd
	 * @param unknown $user_level
	 * @return string|json  success:return json data. failed:return reason
	 */
	public function actionPShowBasicScore($token,$user_id,$user_pwd,$user_level){
		if(trim($token)!=$this->token)
			return 'illegal token';
		if((int)trim($user_level)!=0)
			return 'illegal user';
		$exist = Basket_user::find()->where([
				'user_id'=>(int)trim($user_id),
				'user_password'=>trim($user_pwd),
				'user_level'=>(int)trim($user_level)
		])->exists();
		if($exist)
			return json_encode(Scoring_formula::find()->asArray()->all());
		return 'illegal user';
	}
	
	/**
	 * modify basic score and describing
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $user_pwd
	 * @param unknown $user_level
	 * @param unknown $bs_id
	 * @param unknown $nb_score
	 * @param unknown $desc
	 * @return string|number success:return 1. failed:return reason
	 */
	public function actionPModifyBasicScoreAndDesc($token,$user_id,$user_pwd,$user_level,$bs_id,$nb_score,$desc){
		if(trim($token)!=$this->token)
			return 'illegal token';
		if((int)trim($user_level) != 0)
			return 'illegal user';
		$exist = Basket_user::find()->where([
				'user_id'=>(int)trim($user_id),
				'user_password'=>trim($user_pwd),
				'user_level'=>(int)trim($user_level)
		])->exists();
		if($exist){
			$item = Scoring_formula::findOne($bs_id);
			$item['basic_score']=(int)trim($nb_score);
			$item['describing']=trim($desc);
			if($item->save())
				return 1;
			else 
				return 'validation failed';
		}
	}

	/**
	 * add scoring formular
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $user_pwd
	 * @param unknown $user_level
	 * @param unknown $type  1,ligong  2,sheke 3,art 4,jiaoxueyanjiu
	 * @param unknown $project
	 * @param unknown $content
	 * @param unknown $basic_score
	 * @param unknown $desc
	 * @return string|number success:return 1. failed:return reason
	 */
	public function actionPAddBasicScore($token,$user_id,$user_pwd,$user_level,$type,$project,$content,$basic_score,$desc){
		if(trim($token)!=$this->token)
			return 'illegal token';
		if((int)trim($user_level) != 0)
			return 'illegal user';
		$exist = Basket_user::find()->where([
				'user_id'=>(int)trim($user_id),
				'user_password'=>trim($user_pwd),
				'user_level'=>(int)trim($user_level)
		])->exists();
		if($exist){
			$new_item = new Scoring_formula();
			return $new_item->addItem($type, $project, $content, $basic_score, $desc);
		}else 
			return 'illegal count';	
	}
	
	/**
	 * modify basic multi
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $user_pwd
	 * @param unknown $user_level
	 * @param unknown $num
	 * @return json|number success:return 1. failed:return reason
	 */
	public function actionPModifyBasicMuti($token,$user_id,$user_pwd,$user_level,$num){
		if(trim($token)!=$this->token)
			return 'illegal token';
		if((int)trim($user_level) != 0)
			return 'illegal user';
		$exist = Basket_user::find()->where([
				'user_id'=>(int)trim($user_id),
				'user_password'=>trim($user_pwd),
				'user_level'=>(int)trim($user_level)
		])->exists();
		$item = Influence::findOne(1);
		$item['value']=(int)trim($num);
		if($item->save())
			return 1;
		else 
			return json_encode($item->getErrors());
	}
	
	/**
	 * get basic multi
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $user_pwd
	 * @param unknown $user_level
	 * @return number|string success:return result. failed:return reason
	 */
	public function actionPShowBasicMuti($token,$user_id,$user_pwd,$user_level){
		if(trim($token)!=$this->token)
			return 'illegal token';
		if((int)trim($user_level) != 0)
			return 'illegal user';
		$exist = Basket_user::find()->where([
			'user_id'=>(int)trim($user_id),
			'user_password'=>trim($user_pwd),
			'user_level'=>(int)trim($user_level)
		])->exists();
		if($exist)
			return Influence::findOne(1)['value'];
		else 
			return 'invalid count';
	}
	
	/**
	 * show unpassed achievements and pass it
	 * @param unknown $token
	 * @param unknown $type  0:show  1:pass or not
	 * @param number $a_id
	 * @param number $stauts
	 * @return number|json|string
	 */
	public function actionValidateAchievement($token,$type,$a_id=0,$status=0){
		if(trim($token)!=$this->token)
			return 'illegal token';
		if((int)trim($type)==0){
			$items = Achievement::find()->where(['checked'=>0])->asArray()->all();
			return json_encode($items);
		}else if((int)trim($type)==1){
			$item = Achievement::find()->where(['id'=>(int)trim($a_id)])->one();
			$item['checked']=(int)trim($status);
			if($item->save())
				return 1;
			else 
				return json_encode($item->getErrors());
		}else 
			return 'invalid operation';
		
	}
	/**
	 * get personal information by u_id
	 * @param unknown $u_id
	 */
	public function actionGetPIByUid($u_id){
		$item = Personal_information::find()->where(['u_id'=>trim($u_id)])->asArray()->one();
		return json_encode($item);
	}
	/**
	 * get achievement by u_id
	 * @param unknown $u_id
	 */
	public function actionGetAchiByUid($u_id){
		$items = Achievement::find()->where(['u_id'=>trim($u_id)])->asArray()->all();
		return json_encode($items);
	}
	
	/**
	 * get the information of curriculums
	 * @param unknown $token
	 * @return json|string
	 */
	public function actionGetCurr($token){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$items = Curriculum::find()->asArray()->all();
		return json_encode($items);
	}
	
	/**
	 * modify or add the curriculum
	 * @param unknown $token
	 * @param unknown $new_name
	 * @param unknown $serial_number
	 * @param number $id  add: don't need to pass $id
	 */
	public function actionModifyCurr($token,$new_name,$serial_number,$id=0){
		$item=0;
		if($id===0)
			$item = new Curriculum();
		else 
			$item = Curriculum::findOne((int)trim($id));
		$item['name']=trim($new_name);
		$item['serial_number']=trim($serial_number);
		if($item->save())
			return 1;
		else 
			return json_encode($item->getErrors());
	}
	
	/**
	 * show groups
	 * @param unknown $token
	 * @return json
	 */
	public function actionPShowGroups($token){
		$items = Group::find()->asArray()->all();
		return json_encode($items);
	}
	/**
	 * modify group name
	 * @param unknown $token
	 * @param unknown $id
	 * @param unknown $name
	 * @return number|string|json 1:success
	 */
	public function actionPModifyGroup($token,$id,$name){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$exists = Group::find()->where(['id'=>(int)trim($id)])->exists();
		if($exists){
			$item = Group::findOne((int)trim($id));
			$item['name']=trim($name);
			if($item->save())
				return 1;
			else 
				return json_encode($item->getErrors());
		}
	}
	/**
	 * add group
	 * @param unknown $token
	 * @param unknown $name
	 * @return number|string|json 1:success
	 */
	public function actionPAddGroup($token,$name){
		$exists = Group::find()->where(['name'=>trim($name)])->exists();
		if($exists)
			return 'the group has already exists';
		$item = new Group();
		$item['name']=trim($name);
		if($item->save())
			return 1;
		else  
			return json_encode($item->getErrors());
	}
	
	/**
	 * delete group
	 * @param unknown $token
	 * @param unknown $id
	 * @return number 1:success  0:failed
	 */
	public function actionPDeleteGroup($token,$id){
		$item = Group::findOne((int)trim($id));
		if($item->delete())
			return 1;
		else 
			return 0;
	}
	
	/**
	 * add workload
	 * @param unknown $token
	 * @param unknown $user_id
	 * @param unknown $curriculum_id
	 * @param unknown $date
	 * @param unknown $department
	 * @param unknown $course_character
	 * @param unknown $credit
	 * @param unknown $total_class_hours
	 * @param unknown $theory_class_hours
	 * @param unknown $practice_class_hours
	 * @param unknown $com_class_hours
	 * @param unknown $number_of_students
	 * @param number $number_of_classes
	 * @param number $classes
	 * @param number $k1
	 * @param number $k5
	 * @param number $k6
	 */
	public function actionPAddWorkload($token,$user_id,$curriculum_id,$date,
			$department,$course_character,$credit,$total_class_hours=0,$theory_class_hours=0,
			$practice_class_hours=0,$com_class_hours=0,$number_of_students=0,$number_of_classes=0,
			$classes=0,$k1=0,$k5=0,$k6=0){
		
		if(trim($token)!=$this->token)
			return 'illegal token';
		$item = new TeachExp();
		$item['u_id'] = (int)trim($user_id);
		$item['curriculum_id']=(int)trim($curriculum_id);
		$item['date_begin_end']=trim($date);
		$item['department']=trim($department);
		$item['course_character']=trim($course_character);
		$item['credit']=(int)trim($credit);
		$item['total_class_hours']=(int)trim($total_class_hours);
		$item['theory_class_hours']=(int)trim($theory_class_hours);
		$item['practice_class_hours']=(int)trim($practice_class_hours);
		$item['com_class_hours']=(int)trim($com_class_hours);
		$item['number_of_students']=(int)trim($number_of_students);
		if(trim($course_character=='选修')){
			if((int)trim($number_of_students)%30>15)
				$number_of_classes = (int)$number_of_students/30 + 1;
			else 
				$number_of_classes = (int)$number_of_students/30;
		}
		$item['number_of_classes']=(int)trim($number_of_classes);
		$item['classes']=trim($classes);
		if($item->save()){
			return $this->calFactorDistribution($k1,$k5,$k6);		
		}else 
			return json_encode($item->getErrors());
	}
	
	public function actionPUploadPic(){
		
	}
	public function actionPGetUserDoc($token,$user_id){
		return $this->GenerateDoc($token, $user_id);
	}
	//public function actionSubmitTeach($token,)
	
	/**
	 * email is unique or not
	 * @param unknown $email
	 * @return number 0:existed  1:not existed
	 */
	public function actionUnique($email){
		$exist = Basket_user::find()->where(['user_email'=>trim($email)])->exists();
		if($exist)
			return 0;
		else
			return 1;
	}
	
	/**
	 * get achievements of user by user_id
	 * @param unknown $token
	 * @param unknown $u_id
	 * @return json
	 */
	public function actionPGetAchiByUserId($token,$u_id){
		if(trim($token)  != $this->token)
			return 'illegal token';
		$items = Achievement::find()->where(['u_id'=>trim($u_id)])->asArray()->all();
		return  json_encode($items);
	}
	/**
	 * get all the value of k
	 */
	public function actionGetTeachFactor(){
		return TeachFactor::getAll();
	}
	/**
	 * modify the value of  k
	 * @param unknown $id
	 * @param unknown $value1
	 * @param unknown $value2
	 * @param unknown $value3
	 */
	public function actionModifyTeachFactor($id,$value1,$value2,$value3){
		$id = (int)trim($id);
		$value1 = (float)trim($value1);
		$value2 = (float)trim($value2);
		$value3 = (float)trim($value3);
		$fac = TeachFactor::findOne($id);
		if($fac->updateItem($value1, $value2, $value3))
			return 1;
		return 0;
	}
	
	/**
	 * generate the doc file of specific user
	 * @param unknown $token
	 * @param unknown $user_id
	 * @return txt  
	 */
	private function GenerateDoc($token,$user_id){
		if(trim($token)!=$this->token)
			return 'illegal token';
		$user_info = Personal_information::find()->where(['u_id'=>trim($user_id)])->one();
		$achievements = Achievement::find()->where(['u_id'=>trim($user_id),'checked'=>2])->all();
		$edu_expers = Education_experience::find()->where(['uer_id'=>trim($user_id)])->orderBy(['date_begin'=>SORT_DESC])->all();
		$work_expers = Work_experience::find()->where(['uer_id'=>trim($user_id)])->orderBy(['date_begin'=>SORT_DESC])->all();
		return $this->renderPartial('userdoc',['user_info'=>$user_info,'achievements'=>$achievements,'edu_expers'=>$edu_expers,'work_expers'=>$work_expers]);
	}
	
	/**
	 * cal the factor distribution of workload
	 * @param unknown $k1
	 * @param unknown $k5
	 * @param unknown $k6
	 */
	private function calFactorDistribution($k1,$k5,$k6){
		$item = TeachExp::find()->orderBy(['id'=>SORT_DESC])->one();
		$t_id = $item['id'];
		$item_workload_factor = new WorkloadFactorDistribution();
		$item_workload_factor['t_id'] = $t_id;
		$item_workload_factor['k1']=(float)trim($k1);
		$number = TeachExp::find()->where(['u_id'=>$item['u_id'],'curriculum_id'=>$item['curriculum_id']])->count();
		if($number>1){
			$k2 = TeachFactor::findOne(2);
			$item_workload_factor['k2']=$k2['value_1'];
		}else if($number==1){
			$k2 = TeachFactor::findOne(2);
			$item_workload_factor['k2']=$k2['value_2'];
		}
		$item_workload_factor['k3']=TeachFactor::findOne(3)['value_1']+$item['number_of_classes']*0.2;
		$item_workload_factor['k4']=1.0;
		$item_workload_factor['k5']=(float)trim($k5);
		
		if($item['number_of_classes'] ==1)
			$item_workload_factor['k6']=1.35;
		else if($item['number_of_classes'] > 1)
			$item_workload_factor['k6']=1;	
		
		if($number >1)
			$item_workload_factor['k7']=0.8;
		else if($number == 1)
			$item_workload_factor['k7']=1;
		if($item['com_class_hours']==0){
			$item_workload_factor['k5']=0.0;
			$item_workload_factor['k6']=0.0;
			$item_workload_factor['k7']=0.0;
		}else if($item['theory_class_hours']==0){
			$item_workload_factor['k2']=0.0;
			$item_workload_factor['k3']=0.0;
			$item_workload_factor['k4']=0.0;
		}
		if($item_workload_factor->save()){
			return $this->calWorkloadDistribution($t_id,$item_workload_factor['id']);
		}
		else 
			return 0;	
	}
	
	/**
	 * calWorkloadDistribution
	 * @param unknown $t_id
	 * @param unknown $fac_dis_id
	 */
	private function calWorkloadDistribution($t_id,$fac_dis_id){
		$c_hours = TeachExp::findOne((int)trim($t_id))['com_class_hours'];
		$c_classes = TeachExp::findOne((int)trim($t_id))['number_of_classes'];
		$t_hours = TeachExp::findOne((int)trim($t_id))['theory_class_hours'];
		$p_hours = TeachExp::findOne((int)trim($t_id))['practice_class_hours'];
		$k1=WorkloadFactorDistribution::findOne($fac_dis_id)['k1'];
		$k2=WorkloadFactorDistribution::findOne($fac_dis_id)['k2'];
		$k3=WorkloadFactorDistribution::findOne($fac_dis_id)['k3'];
		$k4=WorkloadFactorDistribution::findOne($fac_dis_id)['k4'];
		$k5=WorkloadFactorDistribution::findOne($fac_dis_id)['k5'];
		$k6=WorkloadFactorDistribution::findOne($fac_dis_id)['k6'];
		$k7=WorkloadFactorDistribution::findOne($fac_dis_id)['k7'];
		$theory = $t_hours*$k1*$k2*$k3*$k4;
		$com = $c_hours*$c_classes*$k1*$k5*$k6*$k7;
		$practice = 0;
		$total = $theory+$com+$practice;
		
		$item = new WorkloadDistribution();
		$item['t_id']=$t_id;
		$item['workload_of_theory']=$theory;
		$item['workload_of_cum']=$com;
		$item['workload_of_practice']=$practice;
		$item['workload_of_total']=$total;
		if($item->save())
			return 1;
		else 
			return 0;
	}

	/**
	 * 1 -> one  such like that
	 * @param unknown $i
	 * @return string 1->one
	 */
	private function intToEng($i){  // 1  转成 one，  2 到 two
		$eng='zero';
		switch($i){
			case 1:
				$eng='one';
				break;
			case 2:
				$eng='two';
				break;
			case 3:
				$eng='three';
				break;
			case 4:
				$eng='four';
				break;
			case 5:
				$eng='five';
				break;
			case 6:
				$eng='six';
				break;
			case 7:
				$eng='seven';
				break;
			default:
				break;
		}
		return $eng;
	}
	/**
	 * calculate the total score of specific user
	 * @param unknown $user_id
	 * @return number return the total score
	 */
	private function calTotalScore($user_id){
		$items = Achievement::find()->where(['user_id'=>(int)trim($user_id),'checked'=>2])->asArray()->all();
		$total_score = 0;
		foreach ($items as $item)
			$total_score+=$item['score'];
		$user = Basket_user::findOne((int)trim($user_id));
		$user['user_total_score']=$total_score;
		if($user->save())
			return 1;
		else 
			return 0;
	}
}
?>