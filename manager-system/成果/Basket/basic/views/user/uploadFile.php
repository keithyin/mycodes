<div class="login">

<h1>Upload Image</h1>

<?php
	use yii\widgets\ActiveForm;
	use yii\helpers\Html;
	use yii\widgets\ActiveField;
	$form = ActiveForm::begin(
			['options'=>['enctype'=>'multipart/form-data','method'=>'post']]);
?>
	<?=$form->field($model, 'imageFile')->fileInput()?>
	<?=Html::submitButton('Submit')?>

<?php ActiveForm::end();?>
</div>
