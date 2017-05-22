<?php
	namespace app\assets;
	use yii\web\AssetBundle;
	use app\assets\JQueryAsset;
	class LoginAsset extends AssetBundle{
		public $basePath = '@webroot';
		public $baseUrl = '@web';
		public $css = ['css/login.css',];
		public $js = ['js/submit_achievement.js'];
		public $depends=[
				'yii\web\YiiAsset',
				'yii\bootstrap\BootstrapAsset',
		];	
	}
?>