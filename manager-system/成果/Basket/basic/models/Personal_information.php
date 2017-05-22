<?php
namespace app\models;
use yii\db\ActiveRecord;

class Personal_information extends ActiveRecord{
	
	public function rules(){
		return [
				[['real_name','sex','birthday','idcard','nation','wordplace','rank','highest_degree','email','telephone_number','u_id'],'required'],
				['birthday','string'],
				['idcard','string','length'=>18],
				['telephone_number','string','length'=>11],		
		];
	}
	public static function tableName(){
		return 'personal_information';
	}
	public function addPersonInfo($u_id){
		$this['u_id']=(int)trim($u_id);
		$this['real_name']='***';
		$this['sex']=0;  //0:man  1:women
		$this['birthday']='1111-11-11';
		$this['idcard'] = 'xxxxxxxxxxxxxxxxxx';
		$this['nation']='族';
		$this['wordplace']='empty';
		$this['rank']='***';
		$this['highest_degree']='***';
		$this['email']='y@hotmail.com';
		$this['telephone_number']='11111111111';
		$this['department']='***';
		$this['group_id']=1;
		if($this->save())
			return 1;
		else 
			return $this->getErrors();
	}
	public function deleteItem(){
		if($this->delete())
			return 1;
		else  
			return $this->getErrors();
	}
	public function modifyItem(){
		
	}
}
?>