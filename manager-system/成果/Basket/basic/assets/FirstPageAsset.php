<?php
	namespace app\assets;
	
	use yii\web\AssetBundle;
	
	class FirstPageAsset extends AssetBundle{
		public $basePath='@webroot';
		public $baseUrl='@web';
		public $css=['css/first_page.css',];
		public $js=[];
		public $depends=[
				'yii\web\YiiAsset',
       	 		'yii\bootstrap\BootstrapAsset',
		];	
	}
?>