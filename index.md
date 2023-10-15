---
layout: base
description: Blog about Programming and Quantitative Finance.
---

<ul class="list-inside list-none">
  {% for post in site.posts %}
    <li class="m-0 py-2">
      <a class="text-xl" href="{{ post.url }}">{{ post.title }}</a>
      <time class="block text-sm post-subtitle">{{ post.date | date: site.my.date_format }}</time>
    </li>
  {% endfor %}
</ul>
