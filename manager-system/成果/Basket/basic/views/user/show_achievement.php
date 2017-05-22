

<div class='login'>
<input id="search_achi" type="text" name="keyword" placeholder="请输入关键字进行查询" />
<button type="button" id="search_achi_button">查询</button>
</div>
<br/>

<div class="achievement">

<h1>My Achievements</h1>

<table id="show_achievement">

<tr><th>Achievement</th><th>Syn_Position</th><th>Score</th><th>Checked</th></tr>
<?php foreach($achievements as $achievement):?>
	<tr>
	<td><?=$achievement['project_name']?></td>
	<td><?=$achievement['syn_position']?></td>
	<td><?=$achievement['score']?></td>
	<td><?php 
			if(0==$achievement['checked'])
				echo "<font color=\"#C93E26\">not yet.</font>";
			else if(1==$achievement['checked'])
				echo "<font color=\"red\"> failed.</font>";
			else 
				echo "<font color=\"green\"> pass.</font>";
		?>
	</td>
	</tr>
<?php endforeach;?>

</table>
</div>