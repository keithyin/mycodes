 <div class="login">
      <h1>Login</h1>
      <?php 
      	use yii\widgets\ActiveForm;
      	use yii\helpers\Html;
      ?>
      <?php $form=ActiveForm::begin([
      		'id'=>'login-form',
      		'options'=>['method'=>'post',],
      		]) ;
      ?>
       <p><?=$form->field($model, 'userMail')->label('mail') ?> </p>
       <p><?=$form->field($model,'password')->passwordInput()->label('password')?> </p>
       <p class="submit"><?=Html::submitButton('Login')?> </p>
        <?php ActiveForm::end();?>
        <p><a href="./index.php?r=user/register">New User?</a> </p>
        
    <div class="login-help">
      <p>forget password? <a href="index.html">click to modify</a>.</p>
    </div>
 </div>

