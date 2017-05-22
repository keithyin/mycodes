<?php
	namespace app\models;
	use yii\db\ActiveRecord;
	
	class Scoring_formula extends ActiveRecord{
		public static function tableName(){
			return 'scoring_formula';
		}
		public function addItem($type,$project,$content,$basic_score,$desc){
			$this->type = (int)trim($type);
			$this->project = trim($project);
			$this->content = trim($content);
			$this->basic_score=(int)trim($basic_score);
			$this->describing = trim($desc);
			if($this->save())
				return 1;
			else  
				return 'validation failed';
		}
	}
?>