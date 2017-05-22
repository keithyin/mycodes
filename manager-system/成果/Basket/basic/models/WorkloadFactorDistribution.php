<?php
namespace app\models;
use yii\db\ActiveRecord;
class WorkloadFactorDistribution extends ActiveRecord{
	public static function tableName(){
		return 'workload_factor_distribution';
	}
	
	public function rules(){
		return [
				['t_id','required'],
		];
	}
}
?>