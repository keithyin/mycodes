<div class="edu">
<h1>Work Experience</h1>

<?php 
	use yii\widgets\ActiveForm;
	use yii\helpers\Html;
	use dosamigos\datepicker\DatePicker;
	$count = 0;
	$form = ActiveForm::begin([
		'id'=>'work_experience',
		'enableAjaxValidation'=>false,
	]);
?>
<?php foreach($res as $index=>$re):?>
<?php $count++;?>
<?=$form->field($re,"[$index]date_begin")->widget(DatePicker::className(),[
		'language'=>'zh-CN',
		'template'=>"{addon}{input}",
		'clientOptions'=>[
				'autoclose'=>true,
				'format'=>'yyyy-mm-dd',
		]
])
?>
<?=$form->field($re,"[$index]date_end")->widget(DatePicker::className(),[
		'language'=>'zh-CN',
		'template'=>'{addon}{input}',
		'clientOptions'=>[
				'autoclose'=>true,
				'format'=>'yyyy-mm-dd',
]
		
])?>
<?=$form->field($re,"[$index]company")?>
<?=$form->field($re,"[$index]department")?>
<?=$form->field($re,"[$index]position")?>
<?php endforeach;?>

<?=Html::submitButton('Submit')?>
<?=Html::button('Add')?>
<?php ActiveForm::end();?>
</div>
<!-- 处理add事件的js代码 ，写到了js文件中-->
<?php 
	$js = <<<JS
		
JS;
	$this->registerJS($js);
?>
