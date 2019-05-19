---
layout: page
title: Random
permalink: /posts/
---

{% comment %}

{% for post in site.posts %}
* {{ post.date | date: "%b %-d, %Y" }}\>\>
  [ {{ post.title }}]({{ post.url | prepend: site.baseurl }})
{% endfor %}

{% endcomment %}

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

## VIM and TMUX Configuration

Check out this [repo](https://github.com/feixh/vim_tmux_cfg) for my vim and tmux configuration.


