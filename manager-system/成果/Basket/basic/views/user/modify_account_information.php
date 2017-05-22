<div class='login'>
<h1>Modify Acount Info</h1>

<?php 
	use yii\helpers\Html;
	use yii\widgets\ActiveForm;
	$form = ActiveForm::begin();
?>
<font size="3">My Email:</font><br/><font color="blue"><?=$user_info['user_email']?></font>
<br/><br/>
<?=$form->field($user_info,'user_nickname')->label('<font  size=\"2\">My Nickname:</font>')?>

<?=$form->field($user_info,'user_tele')->label('<font size=\"2\">My Tele:</font>')?>

<?=$form->field($user_info,'user_password')->label('<font size=\"2\">My Password:</font>')?>
<br/>
<?=Html::submitButton('Confirm')?>

</div>

<?php ActiveForm::end();?>