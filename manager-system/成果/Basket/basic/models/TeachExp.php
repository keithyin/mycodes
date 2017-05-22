<?php

namespace app\models;
use yii\db\ActiveRecord;

class TeachExp extends ActiveRecord{
	public static function tableName(){
		return 'teach';
	}
	public function rules(){
		return [
				[['curriculum_id','u_id','date_begin_end'],'required'],
		];
	}
}
?>