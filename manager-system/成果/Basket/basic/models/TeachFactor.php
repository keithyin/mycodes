<?php
namespace app\models;
use yii\db\ActiveRecord;
class TeachFactor extends ActiveRecord{
	public static function tableName(){
		return 'teach_factor';
	}
	public static function getAll(){
		$all = TeachFactor::find()->asArray()->all();
		return json_encode($all);
	}
	
	public function updateItem($value1,$value2,$value3){
		$this->value1=$value1;
		$this->value2=$value2;
		$this->value3=$value3;
		if($this->save())
			return 1;
		else 
			return 0;
	}
}
?>