<?php
	namespace app\controllers;
	use yii\base\Controller;
	
	class WorkController extends Controller{
		public $layout = 'portraite_related';
		//渲染mainPage
		public function actionMainPage(){
			return $this->renderPartial('mainPage');
		}
		public function actionFrameLeft(){
			return $this->render('frame_left');
		}
		public function actionFrameRight(){
			return $this->renderPartial('frame_right');
		}
	}
?>