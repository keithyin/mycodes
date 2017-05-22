<?php
	namespace app\models;
	use yii\db\ActiveRecord;
	class Education_experience extends ActiveRecord{
		
		public function scenarios(){
			$scenarios = parent::scenarios();
			$scenarios['edu_1'] = ['uer_id','date_begin','date_end','university','department','degree','teacher'];
			return $scenarios;
		}
		
		public function rules(){
			return [
				[['date_begin','date_end','university','department','degree','teacher'],'required','on'=>'edu_1'],	
			];
		}
		public function attributeLabels(){
			return [
				'date_begin'=>'起始时间',
				'date_end'=>'结束时间',
				'university'=>'学校',
				'department'=>'院系',
				'degree'=>'学位',
				'teacher'=>'导师',
			];
		}
		public static function tableName(){
			return 'education_experience';
		}
	}
?>