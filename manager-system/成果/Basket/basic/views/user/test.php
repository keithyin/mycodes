<div class='edu'>
<h1>Education Experience</h1>
<?php
	use dosamigos\datepicker\DatePicker;
	use yii\widgets\ActiveForm;
	use yii\helpers\Html;
	$form = ActiveForm::begin([
			'id'=>'edu-exp',
			'enableAjaxValidation'=>false,
	]);
?>
->1
<?=$form->field($edu_1,'date_begin')->widget(DatePicker::className(),[
//		'inline'=>true,
		'template'=>'{addon}{input}',
		'language'=>'zh-CN',
		'clientOptions'=>[
			'autoclose'=>true,
			'format'=>'yyyy-mm-dd',
],
		
])->label('start time')?>
<?=$form->field($edu_1,'date_end')->widget(DatePicker::className(),[
	//	'inline'=>true,
		'template'=>'{addon}{input}',
		'language'=>'zh-CN',
		'clientOptions'=>[
			'autoclose'=>true,
			'format'=>'yyyy-mm-dd',
],
		
])->label('end time')?>

<?=$form->field($edu_1,'university') ?>
<?=$form->field($edu_1,'department')?>
<?=$form->field($edu_1,'degree')?>
<?=$form->field($edu_1,'teacher')?>

<?=Html::submitButton('Submit')?>
</div>
<?php ActiveForm::end()?>


















