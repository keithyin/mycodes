$(function(){	
	var data;
	$('form select').eq(0).blur(function(e){    //焦点离开，就会用数据填充第二个下拉列表
		e.stopPropagation();
		var department = decodeURIComponent($(this).serialize());// department=1,2,3
		$.ajax({
			type:'get',
			url:'?r=user/get-second-level'+                //会自动定位到自家主机
				'&'+ department +'&callback=?',
			
			success:function(response, status, xhr){
				data = $.parseJSON(response);//返回的值添加到第二个下拉列表里
				$('form select').eq(1).append(function(index, html){
					html='<option value="0"></option>';
					var pre = '';
					data.forEach(function(item, index, array){
						if(data[index]['project']!= pre){
							var temp = '<option value='+data[index]['project']+'>'+data[index]['project']+'</option>';
							html += temp;
							pre = data[index]['project'];
						}
					})
					return html;
				});
			}
		});
	});	
	
	$('form select').eq(0).focus(function(e){   //第一个列表聚焦，  立刻清空第二三框数据
		e.stopPropagation();
		$('form select').eq(1).find('option').remove(); //清空第二节点
		$('form select').eq(2).find('option').remove();//清空第三列表数据
	});
	
	$('form select').eq(1).blur(function(e){   //第二个下拉列表焦点离开事件，填充第三个列表数据
		e.stopPropagation();
		$('form select').eq(2).find('option').remove();  //先清空
		var department = $('form select').eq(0).find('option:selected').val();
		var project = $('form select').eq(1).find('option:selected').text();
		
		if(project=="横向项目"){
			$('#fund').show();
			$('#fund_label1').show();
			$('#fund_label2').show();
			$('input[name=fund]').show();
			$('input[name=cal_type]').attr('value',3);
		}
		if(project=="纵向科研项目"){
			$('#fund').show();
			$('#fund_label1').show();
			$('#fund_label2').show();
			$('input[name=fund]').show();
			$('input[name=cal_type]').attr('value',2);	
		}
		$('form select').eq(2).append(function(index, html){   //第三个下拉列表添加数据
			html='<option value="0"></option>';
			data.forEach(function(item, index, array){
				var temp_data = data[index];
				if(temp_data['type']== department && temp_data['project']==project){
					var temp_html = '<option value='+temp_data['content']+'>'+temp_data['content']+'</option>';
					html += temp_html;
				}
			});
			return html;
		});
	});
	$('form select').eq(1).focus(function(e){  //第二个列表聚焦，清空第三列表信息
		e.stopPropagation();
		$('form select').eq(2).find('option').remove();
	});
	$('form select').eq(2).blur(function(e){  //第三个列表失去焦点，判断是否显示隐藏框
		var content = $('form select').eq(2).find('option:selected').text();
		if (content == '公开发表:SSCI检索系统收录' || content=='公开发表:SCI检索系统收录'
			|| content == 'SSCI_AHCI系统收录'){
			$('input[name=f]').show();
			$('font[id=f]').show();
			$('input[name=cal_type]').attr('value',4);
		}
	});
	$('form select').eq(3).blur(function(e){   //第四个列表失去焦点事件
		e.stopPropagation();
		$('form select').eq(4).find('option').remove();
		var count = $('form select').eq(3).find('option:selected').text();
		$('form select').eq(4).append(function(index, html){
			html='<option value="0"> </option>';
			for(var i=0; i<count; i++){
				var temp = '<option value='+(i+1)+'>'+(i+1)+'</option>'
				html+=temp;
			}
			return html;
		});
	});
	
	//register页面的js代码
	$('.register input').eq(1).blur(function(e){
		e.stopPropagation();
	
		if(''==$.trim($('.register input').eq(1).val())){
			$('.register label').eq(0).css('color','red');
			$('.register input').eq(1).css('border-color','red');
		}	
	});
	
	$('.register input').eq(2).focus(function(e){//邮箱的唯一性验证
		e.stopPropagation();
		var input_email =$.trim($('.register input').eq(1).val());
		
/*
		$.ajax({
			type:'get',
			url:encodeURIComponent('?r=other/is-exist&email='+input_email+'&callback=?'),
			success:function(response,status,xhr){
				if(resopnse){  //true 存在，  false 不存在
					$('#email_exist').show();
					$('button').attr('disabled',true);
					alert(response);
				}
			}
		});
*/
		
		$.post('?r=other/is-exist',{email:input_email},function(response,status,xhr){	
			if(response==1){
				$('.register label').eq(0).css('color','red');
				$('.register input').eq(1).css('border-color','red');
				$('#email_exist').show();
				$('button').attr('disabled',true);	
			}else{
				$('.register input').eq(1).css('border-color','#3C763D');
				$('.register label').eq(0).css('color','#3C763D');
				if(''==$.trim($('.register input').eq(1).val())){
					$('.register label').eq(0).css('color','red');
					$('.register input').eq(1).css('border-color','red');
				}
			}	
		});
		
	});

	
	$('.register input').eq(1).focus(function(e){
		e.stopPropagation();
//		$('.register label').eq(0).css('color','black');
//		$('.register input').eq(1).css('border-color','#c4c4c4');
		$('#email_exist').hide();
		$('button').attr('disabled',false);
	});
	
	//工作经历提交的页面  脚本处理
	var count = 0;
	$.post('?r=other/update-work-experience',null,function(response, status, xhr){
		count = response;
	});
	$('#work_experience button[type=button]').click(function(e){   //询问初始代号
		e.stopPropagation();
		e.preventDefault();
		$('#work_experience input:last').after(function(index,html){
			html='<div>start time<div class="input-group date"><span class="input-group-addon"><i class="glyphicon glyphicon-calendar"></i></span><input type="text" id="work_experience-'+count+'-date_begin" class="form-control" name="Work_experience['+count+'][date_begin]" placeholder="1900-01-23"></div></div>';
			html += '<div>end time<div class="input-group date"><span class="input-group-addon"><i class="glyphicon glyphicon-calendar"></i></span><input type="text" id="work_experience-'+count+'-date_begin" class="form-control" name="Work_experience['+count+'][date_end]" placeholder="1900-01-24"></div></div></div>';
			html+='<div>Company<div><input type="text" id="work_experience-'+count+'-company" class="form-control" name="Work_experience['+count+'][company]" value=""></div</div>';
			html+='<div>Department<input type="text" id="work_experience-'+count+'-department" class="form-control" name="Work_experience['+count+'][department]" value=" "></div></div>';
			html+='<div>Position<div><input type="text" id="work_experience-'+count+'-position" class="form-control" name="Work_experience['+count+'][position]" value=""></div></div>';
			return html;
		});
		count++;
	});
	
	//show achievement页面的模糊搜索脚本处理
	$('#search_achi_button').click(function(e){
		e.stopPropagation();
		var key_word =$.trim( $('#search_achi').val());
		if(""!=key_word){
			var html_cache = '';
			html_cache+="<tr><th>Achievement</th><th>Syn_Position</th><th>Score</th><th>Checked</th></tr>";
			$.post('?r=other/search-for-achievement',{keyword:key_word},function(response,status,xhr){
				var achievements = $.parseJSON(response);
				$.each(achievements,function(index,achievement){
					html_cache+=('<tr>'+
					'<td>'+achievement['brif_desc']+'</td>'+
					'<td>'+achievement['syn_position']+'</td>'+
					'<td>'+achievement['score']+'</td><td>');
				if(0==achievement['checked'])
					html_cache+="<font color=\"#C93E26\">not yet.</font>";
				else if(1==achievement['checked'])
					html_cache+="<font color=\"red\"> failed.</font>";
				else 
					html+= "<font color=\"green\"> pass.</font>";
				html_cache+="</td></tr>";
				
				})
				$('#show_achievement').html(html_cache);    //这个为什么要在里面打印，在外面打印就什么都打印不出来？？？？？？
			});	
		}
	});
	
})
