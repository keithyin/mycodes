<?php
namespace app\models;
use yii\db\ActiveRecord;
class AchiPic extends ActiveRecord{
	public static function tableName(){
		return 'achi_pic';
	}
	public function saveItem($achi_id,$path){
		$this->achi_id = $achi_id;
		$this->path = $path;
		return $this->save();
	}
	
}
?>
