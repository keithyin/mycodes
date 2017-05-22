<div class='edu'>
	<h1>Please select the type carefully</h1>
	<?php 
		use yii\helpers\Html;
		echo Html::beginForm() ;
	?>
	<input type="hidden" name='cal_type' value='1'/>  <!-- 1普通， 2纵向科研项目分值计算，3横向计算，4系统收录计算 -->
	院系:<br/>
	<?=Html::dropDownList('department',null,
			['0'=>' ','1'=>'science','2'=>'social science',
					'3'=>'work of art','4'=>'teching and research'])?><br/>
	项目:<br/>
	<?=Html::dropDownList('project', null,['0'=>' '])?><br/>
	内容:<br/>
	<?=Html::dropDownList('content', null,['0'=>' '])?><br/>
	<font  id="fund_label1">到校经费:</font>
	<?=Html::textInput('fund', null)?>
	<font id="fund_label2">万元<br/></font>
	<font id="f">影响因子:</font>
	<?=Html::textInput('f', null)?><br/>
	多少人参加此项目:<br/>
	<?=Html::dropDownList('count',null,['0'=>' ','1'=>'1', '2'=>'2', '3'=>'3',
			'4'=>'4', '5'=>'5', '6'=>'6', '7'=>'7'
	])?><br/>
	你在此项目中的顺位:<br/>
	<?=Html::dropDownList('syn_position',null,['0'=>' '])?><br/>
	<div>开始时间<div class="input-group date"><span class="input-group-addon"><i class="glyphicon glyphicon-calendar"></i></span><input type="text"  class="form-control" name="date_begin" placeholder="1900-01-23"></div></div>'
	<div>结束时间<div class="input-group date"><span class="input-group-addon"><i class="glyphicon glyphicon-calendar"></i></span><input type="text"  class="form-control" name="date_end" placeholder="1900-01-24"></div></div>'
	项目描述:<br/>
	<?=Html::textarea('project_describe','',['placeholder'=>'论文必须严格按照格式填写，其它成果可以不填'])?><br/><br/>
	<br/>
	队友:<br/>
	<?=Html::textarea('teammate','',['placeholder'=>'名字之间用"，"隔开'])?><br/><br/>
	
	<?=Html::submitButton('提交')?>
	<?=Html::endForm();?>
</div>







