<?php
namespace app\models;

use yii\base\Model;

class LoginModel extends Model{
	public $userMail;
	public $password;
	
	public function rules(){
		return [
				[['userMail','password'],'required'],
				['userMail','email'],
		];
	}
}
?>