<?php
namespace app\models;
use yii\db\ActiveRecord;

class Curriculum extends ActiveRecord{
	public static function tableName(){
		return 'curriculum';
	}
	
	public function rules(){
		return [
				['name','required'],
		];
	}
}

?>