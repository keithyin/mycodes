<?php
namespace app\models;
use yii\db\ActiveRecord;
class Group extends ActiveRecord{
	public static function tableName(){
		return 'group_name';
	}
	public function rules(){
		return [
			['name','required'],	
		];
	}
}
?>