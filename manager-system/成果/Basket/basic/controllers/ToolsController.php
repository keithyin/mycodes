<?php
namespace app\controllers;
use yii\web\Controller;
use app\models\Basket_user;

class ToolsController extends Controller{
	public function actionUnique(){
		$email = \Yii::$app->request->get('email','');
		$exist = Basket_user::find()->where(['user_email'=>$email])->exists();
		if($exist)
			return 0;
			else
				return 1;
	}	
}

?>