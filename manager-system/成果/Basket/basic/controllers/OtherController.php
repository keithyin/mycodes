<?php
namespace app\controllers;
use yii\web\Controller;
use app\models\Basket_user;
use app\models\Education_experience;
use app\models\Work_experience;
use yii\helpers\Url;
use app\models\Achievement;

class OtherController extends Controller{
	private $token='lalala';
 	public $layout = 'loginAndRegister';
 			
 	public function actionSearchAchievement($brif_desc){
 			
 	}
 	//判断传入的邮箱是否存在
 	public function actionIsExist(){
 		$data = \Yii::$app->request->post();
 		$email = $data['email'];
 		$res = Basket_user::find()->where(['user_email'=>$email])->exists();
 		
 		return $res;	
 	}
 	//模糊查找功能,通过教师姓名或成就查找。
 	public function actionSearch(){
 		$session = \Yii::$app->session;
 		$keyword = \Yii::$app->request->post('keyword');
 		$res = Achievement::find()->where(['or',"user_nickname=$keyword",['like','brif_desc',"$keyword"]])->asArray()->all();
 		return json_encode($res);
 		
 	}
 	//添加教育经历
 	public function actionUpdateEducationExperience(){
 		$session = \Yii::$app->session;
 		$user = Basket_user::find()->where(['user_email'=>$session['user_email']])->one();
 		$user_id = $user->user_id;
 		$session['user_id']=$user_id;
 		$edu_1 = new Education_experience(['scenario'=>'edu_1']);
 		$edu_2 = new Education_experience;
 		$edu_3 = new Education_experience;
 		for($i=1; $i <= 3; $i++){         //在这里如果我不使用edu_1场景的话， 数据会存不进去，不知道为什么
 			$edu[$i] = new Education_experience(['scenario'=>'edu_1']);
 		}
 		if(\Yii::$app->request->isPost){
 			$res = \Yii::$app->request->post('Education_experience');
 			for($i=1; $i<=3; $i++){
 				$edu[$i]->attributes = $res[$i];
 				$edu[$i]->uer_id = $user_id;
 			}
 			foreach($edu as $ed)//因为只有第一个model是必填的，所以采取了这种方法进行验证
 				if($ed->validate())
 					$ed->save();
 			return $this->redirect(Url::toRoute('work/main-page'));
 		}else
 			return $this->render('update_education_experience',['edu_1'=>$edu_1,'edu_2'=>$edu_2,'edu_3'=>$edu_3]);
 	}
 	//更新工作经历
 	public function actionUpdateWorkExperience(){
 		$session = \Yii::$app->session;
 		$res_1 = Work_experience::find()->where(['uer_id'=>$session['user_id']])->orderBy('date_begin')->all();
 		$source_count = Work_experience::find()->where(['uer_id'=>$session['user_id']])->orderBy('date_begin')->count();
 		if(\Yii::$app->request->isAjax){
 			return  $source_count;
 		}elseif(\Yii::$app->request->isPost){
 			
// 			$count = count(\Yii::$app->request->post('Work_experience'));
 			$res_2 = \Yii::$app->request->post('Work_experience');
 			
 			foreach($res_2 as $index => $value){
 				if($index < $source_count){
 					$res_1[$index]->date_begin = $value['date_begin'];
 					$res_1[$index]->date_end = $value['date_end'];
 					$res_1[$index]->company=$value['company'];
 					$res_1[$index]->department=$value['department'];
 					$res_1[$index]->position=$value['position'];
 					$res_1[$index]->scenario='submit';	
 				}
 				else {
 					$res_1[$index] = new Work_experience(['scenario'=>'submit']);
 					$res_1[$index]->attributes = $value;
 					$res_1[$index]->uer_id = $session['user_id'];
 				}
 				if($res_1[$index]->validate()){
 					$res_1[$index]->save();
 					
 				}
 			}		
 			exit;
 		}else 
 			return $this->render('update_work_experience',['res'=>$res_1]);
 	}
 	//查找成就功能
 	public function actionSearchForAchievement(){
 		$session = \Yii::$app->session;
 		$user_id = $session['user_id'];
 		$keyword = \Yii::$app->request->post('keyword');
 		$res = Achievement::find()->where(['and',"user_id=$user_id",['like','brif_desc',"$keyword"]])->asArray()->all();
 		return json_encode($res);
 	}
 	//获取所有教师姓名    Phone
 	public function actionGetAllUsers(){   //need token to get the users information
 		if(\Yii::$app->request->isGet){
 			if($this->token == \Yii::$app->request->get('token','')){
 				$users = Basket_user::find()->asArray()->all();
 				return json_encode($users);
 			}else 
 				return 'token error';
 		}else 
 			return 'get out of here.';
 	}
 }
?>
 