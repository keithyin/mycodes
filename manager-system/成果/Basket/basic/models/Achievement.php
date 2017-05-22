<?php
	namespace app\models;
	use yii\db\ActiveRecord;
	
	class Achievement extends ActiveRecord{
		public static function tableName(){
			return 'achievement';
		}
		
		/**
		 * 
		 * @param unknown $user_id
		 * @param unknown $cf_id
		 * @param unknown $syn_position
		 * @param unknown $project_sources
		 * @param unknown $serial_nmber
		 * @param unknown $date_begin
		 * @param unknown $date_end
		 * @param unknown $project_name
		 * @param unknown $score
		 * @param unknown $fund
		 * @param unknown $f
		 * @param unknown $status
		 * @param unknown $pro_desc
		 * @param unknown $teammate
		 */
		public function addAchievement($user_id,$cf_id,$syn_position,$project_sources,$serial_nmber,$date_begin,$date_end,$project_name,$score,$fund,$f,$status,$pro_desc,$teammate){
			$this->u_id = (int)trim($user_id);
			$this->cf_id = (int)trim($cf_id);
			$this->syn_position=(int)trim($syn_position);
			$this->project_sources = trim($project_sources);
			$this->serial_number = trim($serial_nmber);
			$this->project_name = trim($project_name);
			$this->date_begin = trim($date_begin);
			$this->date_end = trim($date_end);
			$this->score =(int)trim($score);
			$this->status = (int)trim($status);
			$this->f = (float)trim($f);
			$this->checked = 0;
			$this->pro_desc = $pro_desc;
			$this->fund = (float)trim($fund);
			$this->teammate = trim($teammate);
			if($this->save())
				return 1;
			else 
				return 'error';
		}
		public static function getAchiByUserId($u_id){
			$items = Achievement::find()->where(['u_id'=>trim($u_id)])->asArray()->all();
			return json_encode($items);
		}
		public static function getAllAchi(){
			$items = Achievement::find()->asArray()->all();
			return json_encode($items);
		}
		public function modifyAchi($u_id,$cf_id,$syn_position,$project_sources,$serial_nmber,$date_begin,$date_end,$project_name,$score,$fund,$f,$status,$pro_desc,$teammate){
			return $this->addAchievement($user_id, $cf_id, $syn_position, $project_sources, $serial_nmber, $date_begin, $date_end, $project_name, $score, $fund, $f, $status, $pro_desc, $teammate);
		}
		public function deleteAchi(){
			$this->delete();
		}
		
	}
?>