---
layout: page
title: Posts
permalink: /Posts/
---

<!--
{% for post in site.posts %}
* {{ post.date | date: "%b %-d, %Y" }}\>\>
  [ {{ post.title }}]({{ post.url | prepend: site.baseurl }})

{% endfor %}

-->

<!-- add post under the folder _posts and then the post would pop up under the random_bits menu -->

<!-- Original Html Page -->
<div class="Blogs">

  <!-- <h1 class="page-heading">Projects</h1> -->

  <ul class="post-list">
    {% for post in site.posts %}
      <li>
        <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>

        <h2>
          <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
        </h2>
      </li>
    {% endfor %}
  </ul>

  <!--
  <p class="rss-subscribe">subscribe <a href="{{ "/feed.xml" | prepend: site.baseurl }}">via RSS</a></p>
-->


</div>

## VIM Configuration

1. Configure vim: Put the `.vimrc` [file]({{ site.url }}/assets/vimrc) in your home folder and name it as `.vimrc`.
2. Create a folder to hold auto-load scripts: `mkdir -p ~/.vim/autoload`.
3. Install a minimal plugin manager: Put the `plug.vim` [script]({{ site.url }}/assets/plug.vim) in the autoload folder. 
4. Install plugins: `vim +PlugInstall`.

## TMUX Configuration

Put this [file]( {{ site.url }}/assets/tmux.conf) in your home folder and name it as `.tmux.conf`.

