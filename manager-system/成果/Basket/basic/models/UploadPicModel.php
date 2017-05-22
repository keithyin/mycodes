<?php 
	namespace app\models;
	use yii\base\Model;
	use yii\web\UploadedFile;
	use yii\helpers\FileHelper;
	use app\models\AchiPic;
	class UploadPicModel extends Model{
		public $imageFile;    // UploadedFile 类型的
		public $userEmail;
		
		public function scenarios(){
			return [
				'upload_head'=>['imageFile'],
				'upload_pic' =>['imageFile'],
			];
		}
		
		public function rules(){
			return [
				[['imageFile'],'file','skipOnEmpty'=>false,'extensions'=>'png,jpg']	
			];
		}
		
		/**
		 * 上传图片， 并将 图片路径信息 增加到 账户信息的 head_portrait 字段
		 */
		
		public function upload($ach_id){
			if($this->validate()){
				$f_h = new FileHelper();
				$f_h->createDirectory("uploads/$this->userEmail/picture");
				$this->imageFile->saveAs("uploads/$this->userEmail/picture/". 
						$this->imageFile->baseName.'.'.$this->imageFile->extension);
				$pro_pic_path = "uploads/$this->userEmail/picture/". 
						$this->imageFile->baseName.'.'.$this->imageFile->extension;
				// 更新用户数据
				$achi_pic = new AchiPic();
				$achi_pic->saveItem($ach_id, $pro_pic_path);
			}else{
				
				echo 'pic upload failed'; echo '<br/>';
				print_r($this->getErrors());
			}
		}
}	
?>