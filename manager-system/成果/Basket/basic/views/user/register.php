<div class="register">

<h1>Register</h1>
<?php 
	use yii\helpers\Html;
	use yii\widgets\ActiveForm;
	$form = ActiveForm::begin([
			'id'=>'register-form',
			'options'=>['method'=>'post',],
	]);
?>
<p> <?=$form->field($model,'user_email')?> </p>
<font id='email_exist' style='display: none' color='red'>The email has already exits!</font>
<p> <?=$form->field($model,'user_password')->passwordInput()?> </p>
<p><?=$form->field($model,'user_password_repeat')->passwordInput()?></p>
<p> <?=$form->field($model,'user_nickname')?> </p>
<p> <?=$form->field($model,'user_tele')?> </p>
<p><?=$form->field($model, 'user_idcard')->textInput()?></p>
<p><?=$form->field($model, 'user_position')->textInput(['placeholder'=>'学校，院系，职位'])?></p>
<p class="submit"><?=Html::submitButton('Next')?> </p>
<?php ActiveForm::end();?>

</div>