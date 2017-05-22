<?php
	namespace app\models;
	use yii\base\Model;
	class RegisterModel extends Model{
		private $user_email;
		private $user_tele;
		private $user_password;
		private $user_password_repeat;
		private $user_nickname;
		private $user_idcard;
		private $user_position;
		
		public function setUser_email($value){
			$this->user_email = trim($value);
		}
		public function getUser_email(){
			return $this->user_email;
		}
		public function setUser_tele($value){
			$this->user_tele = trim($value);
		}
		public function getUser_tele(){
			return $this->user_tele;
		}
		public function setUser_password($value){
			$this->user_password = trim($value);
		}
		public function getUser_password(){
			return $this->user_password;
		}
		public function setuser_password_repeat($value){
			$this->user_password_repeat = trim($value);
		}
		public function getuser_password_repeat(){
			return $this->user_password_repeat;
		}
		public function setUser_nickname($value){
			$this->user_nickname = $value;
		}
		public function getUser_nickname(){
			return $this->user_nickname;
		}
		public function setUser_idcard($value){
			$this->user_idcard = trim($value);
		}
		public function getUser_idcard(){
			return $this->user_idcard;
		}
		public function setUser_position($value){
			$this->user_position = trim($value);
		}
		public function getUser_position(){
			return $this->user_position;
		}
		
		public function scenarios(){
			$scenarios = parent::scenarios();
			$scenarios['first'] = ['user_email','user_tele','user_password','user_password_repeat','user_nickname','user_idcard','user_position'];
			return $scenarios;
		}
		public function rules(){
			return [
				[['user_email','user_tele','user_password','user_password_repeat','user_nickname','user_idcard','user_position'],'required','message'=>'不能为空'],
				['user_password_repeat','compare','compareAttribute'=>'user_password','message'=>'两次密码要一致'],
				['user_email','email'],
				['user_password','string','min'=>6,'max'=>20],
				
			];
		}
		public function attributeLabels(){
			return [
				'user_email'=>'邮箱',
				'user_password'=>'密码',
				'user_password_repeat'=>'确认密码',
				'user_tele'=>'手机号',
				'user_nickname'=>'姓名',
				'user_idcard'=>'身份证号',
				'user_position'=>'职位',
			];
		}
	}
	
?>