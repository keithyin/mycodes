<?php
namespace app\models;
use yii\db\ActiveRecord;
use app\models\Achievement;
class Basket_user extends ActiveRecord{
	private $re_password;
	public function setRe_password($value){
		$this->re_password=$value;
	}
	public function getRe_password(){
		return $this->re_password;
	}
	public function scenarios(){
		$scenarios = parent::scenarios();
		$scenarios['register'] = ['user_email','user_password','re_password','user_tele','user_nickname'];
		$scenarios['modify'] = ['user_password','user_tele', 'user_nickname'];
		$scenarios['p_register']=['user_email','user_password','user_tele','user_nickname'];
		return $scenarios;
	
	}
	
	public function rules(){
		return [
				[['user_email','user_password','user_tele','user_nickname'],'required'],
				['user_email','email'],	
				['user_password','string','min'=>6,'max'=>20],
				['user_email','unique','targetClass'=>'\app\models\Basket_user'],
				['re_password','compare','compareValue'=>$this->user_password, 'on'=>['register']],
		];
	}

	public function attributeLabels(){
		return [
			'user_email'=> 'your email',
			'user_password'=>'your password',
			'user_tele' => 'your telephone number',
			'user_nickname'=> 'your nickname',	
			're_password'=>'repeat your password',
		];
	}
	public static function tableName(){
		return 'basket_user';
	}
	
	//关联查询   建立联系的函数，  查询结果  用 Basket_user::achievements 属性获取
	public function getAchievements(){
		return $this->hasMany(Achievement::className(), ['u_id'=>'user_id']);
	}
	
	public function addUser($email,$pwd,$tele,$nickname){
		$this->user_email = $email;
		$this->user_password=$pwd;
		$this->user_tele=$tele;
		$this->user_nickname=$nickname;
		$this->user_level=3;
		return $this->save();
	}
	public function deleteUser(){
		return $this->delete();
	}
	public function modifyCount($nickname,$tele){
		$this->user_nickname=trim(trim($nickname));
		$this->user_tele=trim($tele);
		if($this->save())
			return 1;
		else 
			return json_encode($this->getErrors());
	}
	public function modifyPwd($new_pwd){
		$this->user_password = trim($new_pwd);
		if($this->save())
			return 1;
		else 
			return json_encode($this->getErrors());
	}
	public static function getAllUsers(){
		$items = Basket_user::find()->asArray()->all();
		return json_encode($items);
	}	
}
?>