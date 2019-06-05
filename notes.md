---
layout: page
title: Notes
permalink: /notes/
---

- [VIM and TMUX Configuration](https://github.com/feixh/vim_tmux_cfg)

{% comment %}
- [Deep Learning - Depth Estimation 1](oldnotes/depth_estimation1.html)
- [Deep Learning - Depth Estimation 2](oldnotes/depth_estimation2.html)
- [Deep Learning - Geometry](oldnotes/deep_geometric_vision.html)
- [Deep Learning - Pixel-wise Prediction](oldnotes/pixelwise_prediction_architecture.html)
- [Deep Learning - Tracking](oldnotes/tracking.html)
- [MCMC](oldnotes/MCMC.html)
- [Model-based Tracking](oldnotes/model_based_tracking.html)
- [SLAM](oldnotes/slam.html)
{% endcomment %}


{% comment %}
{% for post in site.posts %}
* {{ post.date | date: "%b %-d, %Y" }}\>\>
  [ {{ post.title }}]({{ post.url | prepend: site.baseurl }})
{% endfor %}

<!-- Original Html Page -->
{% comment %}
<h1 class="page-heading">Projects</h1>
{% endcomment %}

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

{% comment %}
<p class="rss-subscribe">subscribe <a href="{{ "/feed.xml" | prepend: site.baseurl }}">via RSS</a></p>
{% endcomment %}


{% endcomment %}
