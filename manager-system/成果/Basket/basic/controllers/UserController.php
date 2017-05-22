<?php
/*
 * 用户控制器，用于进行用户登陆， 用户注册主要功能
 * 
 */
namespace app\controllers;
use yii\web\Controller;
use app\models\LoginModel;
use app\models\Basket_user;
use app\models\UploadPicModel;
use yii\web\UploadedFile;
use app\models\Achievement;
use app\models\Scoring_formula;
use app\models\Distribution_standard;
use app\models\Influence;
use app\models\Education_experience;
use yii\helpers\Url;
use app\models\RegisterModel;
use app\models\Personal_information;
class UserController extends Controller{
	public $layout = 'loginAndRegister';
	
	/*
	 * 实现用户登录的功能
	 */
	public function actionLogin(){
		/*
		 * 打开session
		 */
		$session = \Yii::$app->session;
		if(!$session->isActive)
			$session->open();
		
		$loginModel = new LoginModel;
		if(\Yii::$app->request->isPost){
			$loginModel->attributes = \Yii::$app->request->post('LoginModel');
			
			$res = Basket_user::find()->where([
					'user_email'=>$loginModel->userMail,
					'user_password'=>$loginModel->password,
					
			])->asArray()->one();
			/*
			 * 使用更友好的提示方法，提示账号密码错误，或账号不存在，账号不存在的时候，
			 * 直接跳到注册页面
			 */
			if(!empty($res)){              
				$session['user_email'] = $loginModel->userMail;
				$session['user_nickname']=$res['user_nickname'];
				$session['user_id'] = $res['user_id'];
				
				return $this->redirect('./index.php?r=work/main-page');
			}else{
				return false;
			}
			exit;
		}

		return $this->render('login',['model'=>$loginModel]);
		
	}
	
	
	
	/**
	 * 实现用户注册的功能
	 */
	public function actionRegister(){
		
		$register_model = new Basket_user(['scenario'=>'register']);
		$register = new RegisterModel(['scenario'=>'first']);
		
		if(\Yii::$app->request->isPost ){
			$register_model->attributes = \Yii::$app->request->post('RegisterModel');
			if($register_model->validate()){
				$register_model->user_level = 3;
				$register_model->head_portrait = ' ';
				if($register_model->save()){
					$session = \Yii::$app->session;
					if(!$session->isActive)
						$session->open();
					$session['user_email'] = $register_model->user_email;
					
					return $this->redirect(Url::toRoute('work/main-page'));
				}else
					echo "failed!";
			}
					
		}else if(\Yii::$app->request->isAjax){
			echo 'ajax';
		}else
			return $this->render('register',['model'=>$register]);
			
		}
		
		
	
	
	//显示用户基本信息
	public function actionShowUserInformation(){
		$session = \Yii::$app->session;
		$user_model = new Basket_user;
		$user_model= Basket_user::findOne($session['user_id']);
		
		return $this->render('my_information',['model'=>$user_model]);
	}
	//显示用户的成就信息
	public function actionShowUserAchievement(){
		$session = \Yii::$app->session;
		$res = Basket_user::findOne($session['user_id']);
		$achievements = $res->achievements;
		return $this->render('show_achievement',['achievements'=>$achievements]);
	}
	
	//更改用户基本信息页面
	public function actionModifyAcountInformation(){
		$session = \Yii::$app->session;
		
		$user_info = Basket_user::findOne($session['user_id']);
		$user_info->scenario='modify';
		if(\Yii::$app->request->isPost){
			$changed_info = Basket_user::findOne($session['user_id']);
			//不同的覆盖，  没有的保持原值
			$changed_info->attributes = \Yii::$app->request->post('Basket_user');
			if($changed_info->save())
				return $this->renderPartial('modify_success');
			else
				return $this->renderPartial('modify_failed');	
		}
		return $this->render('modify_account_information',['user_info'=>$user_info]);
	}
	
	//提交最近的成果
	public function actionSubmitTheLatestAchievements(){
		if(\Yii::$app->request->isPost){
			$data = \Yii::$app->request->post();
			$score = $this->calScore($data);
			$this->saveAchievement($data, $score);
			return $this->actionUploadPic();
		}else {
			return $this->render('submit_the_latest_achievements');
		}
	}
	
	public function actionGetSecondLevel($department){
		$res = Scoring_formula::find()->where(['type'=>$department])->asArray()->all();
		return json_encode($res);
		
	}
	
	private function calScore($data){
		$final_score=0;
		$item = Scoring_formula::find()->where([
				'type'=>$data['department'],
				'project'=>$data['project'],
				'content'=>$data['content']
		])->one();
		$session = \Yii::$app->session;
		if(!$session->isActive)
			$session->open();
		$session['cf_id']=$item['id'];
		$basic_score = $item['basic_score'];	
		$percent_item = Distribution_standard::findOne($data['count']);
		$persent = $percent_item[$this->intToEng($data['syn_position'])];
		$base_multi = Influence::findOne(1)['value'];
		switch($data['cal_type']){
			case 1:  //普通的分值计算
				$final_score = $basic_score*$persent;
				break;
			case 2:
				$final_score = $basic_score+($data['fund']-$basic_score/$base_multi)*$base_multi;
				break;
			case 3:
				$final_score = $data['fund']*$basic_score;
				break;
			case 4:
				$final_score = $basic_score + $data['f']*$base_multi;
				break;
			default:
				break;
		}
		$final_score*=$persent;
		return $final_score;	
	}
	private function saveAchievement($data,$score){
		$session = \Yii::$app->session;
		if(!$session->isActive)
			$session->open();
		$achieve = new Achievement();
		$session = \Yii::$app->session;
		$desc = $data['department'].'|'.$data['project'].'|'.$data['content']
			.'|经费:'.$data['fund'].'|因子:'.$data['f'].'|人数:'.$data['count']
			.'|顺位:'.$data['syn_position'].'|描述:'.$data['project_describe'];
		$achieve->u_id = $session['user_id'];
		$achieve->cf_id = $session['cf_id'];
		$achieve->syn_position = $data['syn_position'];
		$achieve->score = $score;
		$achieve->pro_desc = $data['project_describe'];
		$achieve->teammate = $data['teammate'];
		$achieve->save();
		$session['achi_id']=$achieve['id'];
		return;
	}
	public function actionUploadPic(){
		$model = new UploadPicModel(['scenario'=>'upload_pic']);
		if (\Yii::$app->request->isPost) {
			$session=\Yii::$app->session;
			if($session->isActive)
				$session->open();
			$model->imageFile = UploadedFile::getInstance($model, 'imageFile');
			if ($model->upload($session['achi_id'])) {
				// 文件上传成功
				return;
			}
		}
		return $this->render('upload_pic', ['model' => $model]);
	}
	public function actionTest($email='fasf@hfa.com',$pwd='qqqwwwq',$name='yinpeng',$tele='11111111111'){
		$item = new Basket_user();
		$item['user_email']=$email;
		$item['user_password']=$pwd;
		$item['user_nickname']=$name;
		$item['user_tele']=$tele;
		if($item->save())
			return $item['user_id'];
		else 
			return 'error';
	}
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
}
?>