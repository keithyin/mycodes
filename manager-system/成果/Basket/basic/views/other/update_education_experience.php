<div class='edu'>
<h1>受教育经历</h1>
<?php
	use dosamigos\datepicker\DatePicker;
	use yii\widgets\ActiveForm;
	use yii\helpers\Html;
	$form = ActiveForm::begin([
			'id'=>'edu-exp',
			'enableAjaxValidation'=>false,
	]);
?>

<?=$form->field($edu_1,'[1]date_begin')->widget(DatePicker::className(),[
//		'inline'=>true,
		'template'=>'{addon}{input}',
		'language'=>'zh-CN',
		'clientOptions'=>[
			'autoclose'=>true,
			'format'=>'yyyy-mm-dd',
],
		
])?>
<?=$form->field($edu_1,'[1]date_end')->widget(DatePicker::className(),[
	//	'inline'=>true,
		'template'=>'{addon}{input}',
		'language'=>'zh-CN',
		'clientOptions'=>[
			'autoclose'=>true,
			'format'=>'yyyy-mm-dd',
],
		
])?>

<?=$form->field($edu_1,'[1]university') ?>
<?=$form->field($edu_1,'[1]department')?>
<?=$form->field($edu_1,'[1]degree')?>
<?=$form->field($edu_1,'[1]teacher')?>



<?=$form->field($edu_2,'[2]date_begin')->widget(DatePicker::className(),[
//		'inline'=>true,
		'template'=>'{addon}{input}',
		'language'=>'zh-CN',
		'clientOptions'=>[
			'autoclose'=>true,
			'format'=>'yyyy-mm-dd',
],
		
])?>
<?=$form->field($edu_2,'[2]date_end')->widget(DatePicker::className(),[
	//	'inline'=>true,
		'template'=>'{addon}{input}',
		'language'=>'zh-CN',
		'clientOptions'=>[
			'autoclose'=>true,
			'format'=>'yyyy-mm-dd',
],
		
])?>

<?=$form->field($edu_2,'[2]university') ?>
<?=$form->field($edu_2,'[2]department')?>
<?=$form->field($edu_2,'[2]degree')?>
<?=$form->field($edu_2,'[2]teacher')?>



<?=$form->field($edu_3,'[3]date_begin')->widget(DatePicker::className(),[
//		'inline'=>true,
		'template'=>'{addon}{input}',
		'language'=>'zh-CN',
		'clientOptions'=>[
			'autoclose'=>true,
			'format'=>'yyyy-mm-dd',
],
		
])?>
<?=$form->field($edu_3,'[3]date_end')->widget(DatePicker::className(),[
	//	'inline'=>true,
		'template'=>'{addon}{input}',
		'language'=>'zh-CN',
		'clientOptions'=>[
			'autoclose'=>true,
			'format'=>'yyyy-mm-dd',
],
		
])?>

<?=$form->field($edu_3,'[3]university') ?>
<?=$form->field($edu_3,'[3]department')?>
<?=$form->field($edu_3,'[3]degree')?>
<?=$form->field($edu_3,'[3]teacher')?>

<?=Html::submitButton('Submit')?>
<?php ActiveForm::end();?>
</div>

