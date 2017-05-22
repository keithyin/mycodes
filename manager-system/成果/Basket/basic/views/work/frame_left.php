	<?php $session=\Yii::$app->session;?>
	<div class="head_portrait">
	<image src="<?="uploads/default.jpg"?>" />
	<font size="3"><b>Welcome back:  </b></font><?=$session['user_nickname']?>
	</div>
	<table>
	<tr><td><a href="./index.php?r=user/show-user-information" target="right"><font size="3" color="white"><b>My Information</b></font></a></td></tr>
	<tr><td><a href="./index.php?r=user/show-user-achievement" target="right"><font size="3" color="white"><b>My Achievement</b></font></a></td></tr>
	<tr><td><b>用户信息</b></td></tr>
	<tr><td><b>用户信息</b></td></tr>
<!-- <tr><td><a href="./index.php?r=other/update-education-experience" target="right"><font size="3" color="white"><b>Update Education Experience</b></font></a></td></tr>  -->
	<tr><td><a href="./index.php?r=other/update-work-experience" target="right"><font size="3" color="white"><b>Update Work Experience</b></font></a></td></tr>
	<tr><td><a href="./index.php?r=other/search" target="right"><font size="3" color="white"><b>Search </b></font></a></td></tr>
	<tr><td><a href="./index.php?r=user/submit-the-latest-achievements" target="right"><font size="3" color="white"><b>Submit The Latest Achivements </b></font></a></td></tr>
	<tr><td><a href="./index.php?r=user/modify-acount-information" target="right"><font size="3" color="white"><b>Modify Acount Information</b></font></a></td></tr>
	<tr><td><a href="./index.php?r=user/test" target="right"><font size="3" color="white"><b>Test</b></font></a></td></tr>
	
	</table>

