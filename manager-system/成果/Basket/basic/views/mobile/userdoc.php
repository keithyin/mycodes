<!DOCTYPE html>
<!--[if lt IE 7]> <html class="lt-ie9 lt-ie8 lt-ie7" lang="en"> <![endif]-->
<!--[if IE 7]> <html class="lt-ie9 lt-ie8" lang="en"> <![endif]-->
<!--[if IE 8]> <html class="lt-ie9" lang="en"> <![endif]-->
<!--[if gt IE 8]><!--> <html lang="en"> <!--<![endif]-->
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
   <link rel="stylesheet" href="css/userdoc.css" type="text/css" />
  <title>Resume</title>
  <!--[if lt IE 9]><script src="//html5shim.googlecode.com/svn/trunk/html5.js"></script><![endif]-->
</head>
<body>
	<b><font class='xiaoer'><?=$user_info['real_name']?>简历</font><br/>
	<font class='si'><?=$user_info['wordplace']?>,<?=$user_info['department']?>,<?=$user_info['rank']?></font>
	<br/>
	<b><font class='si'>教育经历（从大学本科开始，按时间倒排序）：</font></b>
	<br/>
	<?php 
	foreach ($edu_expers as $item){
		echo "&nbsp&nbsp&nbsp<font class='xiaosi'>".$item['date_begin'].'-'.$item['date_end'].'，';
		echo $item['university'].'，';
		echo $item['department'].'，';
		echo $item['degree'].'，';
		echo "导师：";
		echo $item['teacher'];
		echo "</font>";
	}
	?>
	<br/>
	<b><font class='si'>工作经历（科研与学术工作经理，按时间倒排序）</font></b>
	<br/>
	<?php 
	foreach ($work_expers as $item){
		echo "&nbsp&nbsp&nbsp<font class='xiaosi'>".$item['date_begin'].'-'.$item['date_end'].'，';
		echo $item['company'].'，';
		echo $item['department'].'，';
		echo $item['position'];
		echo "</font>";
	}
	?>
	<br/>
	<b><font class='si'>曾使用证件信息（限3个）</font></b>
	<br/>
	<br/>
	<b><font class='si'>主持或参加科研项目及人才计划项目情况（按时间倒排序）：</font></b>
	<br/>
	<?php 
	foreach($achievements as $index=>$item){
		if($item['project']!='学术论文'&&$item['project']!='专利'&&$item['project']!='论文'&&$item['project']!='著作'){
			echo "&nbsp&nbsp&nbsp<font class='xiaosi'>".($index+1).'、';
			echo $item['project_sources'].'，'.$item['serial_number'].'，'.$item['project_name'].'，';
			echo $item['date_begin'].'-'.$item['date_end'].'，';
			echo $item['fund'].'万，';
			if($item['status'])
				echo "已结题，";
			else 
				echo "在研，";
			if($item['syn_position']==1)
				echo "主持。";
			else 
				echo "参加。";
			echo "</font>";
		}
	}
	?>
	<br/>
	<b><font class='si'>发表论文情况：</font></b>
	<?php 
	foreach($achievements as $index=>$item){
		if($item['project']=='学术论文'||$item['project']=='专利'||$item['project']=='论文'||$item['project']=='著作'){
			echo "&nbsp&nbsp&nbsp<font class='xiaosi'>".($index+1).'、';
			echo $item['pro_desc'];
			echo "</font>";
		}
	}
	?>
	<br/>
	<br/>
	<br/>
	<?php 
	$bir = 0;
	$birs = explode("-",$user_info['birthday']);
	$bir = $birs[0].'年'.$birs[1].'月'.$birs[2].'日';
	?>
	<table>
	<tr><td>姓名：</td><td><?=$user_info['real_name']?></td><td>职称：</td><td><?=$user_info['rank']?></td></tr>
	<tr><td>性别：</td><td><?php if($user_info['sex'])echo '女';else echo '男';?></td><td>最高学位：</td><td><?=$user_info['highest_degree']?></td></tr>
	<tr><td>出生日期：</td><td><?=$bir?></td><td>电子邮箱：</td><td><?=$user_info['email']?></td></tr>
	<tr><td>证件号码：</td><td><?=$user_info['idcard']?></td><td>工作电话：</td><td><?=$user_info['telephone_number']?></td></tr>
	<tr><td>民族：</td><td><?=$user_info['nation']?></td><td>项目分工：</td><td></td></tr>
	<tr><td>单位：</td><td></td><td>每年工作几个月</td><td></td></tr>
	</table>
	
</body>
</html>