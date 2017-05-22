<?php
	namespace app\models;
	use yii\db\ActiveRecord;
	class Work_experience extends ActiveRecord{
		
		public function scenarios(){
			$scenarios = parent::scenarios();
			$scenarios['submit']=['uer_id','date_begin','date_end','company','department','position'];
			return $scenarios;	
		}
		
		public function rules(){
			return [
					[['date_begin','date_end','company','department','position']
							,'required','on'=>'submit'],
			];
		}
		public function attributeLabels(){
			return [
					'date_begin'=>'start time',
					'date_end'=>'end time',
			];
		}
	}
?>