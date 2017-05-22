<?php
namespace app\models;
use yii\db\ActiveRecord;

class WorkloadDistribution extends ActiveRecord{
	public static function tableName(){
		return 'workload_distribution';
	}
	
	public function rules(){
		return [
				['t_id','required'],
		];
	}
}
?>