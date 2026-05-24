---
layout: default
title: Deep Reinforcement Learning - A course on Barto Sutton's book
---

<style>
  .tag-item {
    cursor: pointer !important;
    transition: all 0.15s ease-in-out;
    user-select: none;
    border: 1px solid #e2e8f0;
  }
  .tag-item:hover {
    background-color: #f1f5f9;
    border-color: #cbd5e1;
    color: #0f172a;
  }
  .tag-item.active {
    background-color: #2563eb !important;
    color: white !important;
    border-color: #1d4ed8 !important;
    font-weight: 600;
  }
  .hidden-row {
    display: none !important;
  }
  #lectureBody tr:nth-child(even) {
    background-color: #fcfcfd;
  }
</style>

<div class="w-full flex flex-col xl:flex-row gap-6">

  <!-- TABLE -->
  <div class="flex-1 min-w-0">
    <div class="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse table-auto">
          <thead>
            <tr class="bg-slate-50/30 border-b border-slate-200">
              <th class="px-4 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest w-10 text-center">#</th>
              <th class="px-4 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Title</th>
              <th class="px-4 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Questions</th>
              <th class="px-4 py-3 text-[10px] font-extrabold text-slate-500 uppercase tracking-widest">Solutions</th>
            </tr>
          </thead>
          <tbody id="lectureBody">
            {% assign lectures = site.pages | where: "layout", "post" | where: "category", "Lectures" %}
            {% for p in lectures %}
              {% assign filename = p.path | split: "/" | last | remove: ".md" %}
              {% assign parts = filename | split: "-" %}
              {% assign lecture_num = parts[0] | remove: "lecture" %}
              {% assign lecture_file = p.path | split: "/" | last %}
              {% assign lecture_dir = p.path | remove: lecture_file %}
              {% assign q_prefix = lecture_dir | append: "assets/questions/" %}
              {% assign s_prefix = lecture_dir | append: "assets/questions/solutions/" %}
              {% assign first_delivery = p.deliveries | first | default: "" %}
              
              <tr class="lecture-row border-b border-slate-100 hover:bg-blue-50/20 transition-colors" 
                  data-date="{{ first_delivery }}"
                  data-tags="{{ p.tags | join: ',' }}"
                  data-num="{{ lecture_num }}">
                <td class="px-4 py-3 text-sm font-black text-slate-400 text-center">{{ lecture_num }}</td>
                <td class="px-4 py-3">
                  <a href="{{ p.url | relative_url }}" class="text-sm font-bold text-slate-800 hover:text-blue-600 transition-colors">{{ p.title }}</a>
                </td>
                <td class="px-4 py-3">
                  <div class="flex flex-wrap gap-1.5 items-center">
                    {% assign found_questions = false %}
                    {% for q in site.pages %}
                      {% if q.layout == "post" and q.path contains q_prefix %}
                        {% unless q.path contains s_prefix %}
                          {% assign q_ext = q.path | split: "." | last %}
                          {% if q_ext == "md" %}
                            {% assign found_questions = true %}
                            {% assign q_name = q.path | split: "/" | last | remove: ".md" %}
                            <a href="{{ q.url | relative_url }}" class="text-[10px] font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded hover:bg-emerald-100 transition-colors no-underline border border-emerald-100">{{ q_name }}</a>
                          {% endif %}
                        {% endunless %}
                      {% endif %}
                    {% endfor %}
                    {% unless found_questions %}<span class="text-slate-300 text-[10px]">—</span>{% endunless %}
                  </div>
                </td>
                <td class="px-4 py-3">
                  <div class="flex flex-wrap gap-1.5 items-center">
                    {% assign found_solutions = false %}
                    {% for s in site.pages %}
                      {% if s.layout == "post" and s.path contains s_prefix %}
                        {% assign s_ext = s.path | split: "." | last %}
                        {% if s_ext == "md" %}
                          {% assign found_solutions = true %}
                          {% assign s_name = s.path | split: "/" | last | remove: ".md" %}
                          <a href="{{ s.url | relative_url }}" class="text-[10px] font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded hover:bg-amber-100 transition-colors no-underline border border-amber-100">{{ s_name }}</a>
                        {% endif %}
                      {% endif %}
                    {% endfor %}
                    {% unless found_solutions %}<span class="text-slate-300 text-[10px]">—</span>{% endunless %}
                  </div>
                </td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- RIGHT PANEL: Tags -->
  <aside class="w-full xl:w-56 flex-shrink-0">
    <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-4">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-[10px] font-extrabold text-slate-500 uppercase tracking-widest m-0">Tags</h3>
        <button onclick="clearFilters()" class="text-[9px] font-bold text-blue-600 hover:text-blue-800 uppercase tracking-widest transition-colors">Reset</button>
      </div>
      <div class="flex flex-wrap gap-1.5">
        {% assign tags = "" | split: "," %}
        {% assign post_pages = site.pages | where: "layout", "post" %}
        {% for p in post_pages %}
          {% if p.tags %}
            {% for tag in p.tags %}
              {% unless tags contains tag %}
                {% assign tags = tags | push: tag %}
              {% endunless %}
            {% endfor %}
          {% endif %}
        {% endfor %}
        {% assign tags = tags | sort %}
        {% for tag in tags %}
          <span class="tag-item px-2 py-1 bg-slate-50 text-slate-600 rounded-md text-[10px] font-medium" onclick="filterByTag('{{ tag }}', this)">{{ tag }}</span>
        {% endfor %}
      </div>
    </div>
  </aside>

</div>

<script>
  let currentTag = null;

  function filterByTag(tag, el) {
    if (currentTag === tag) {
      currentTag = null;
      document.querySelectorAll('.tag-item').forEach(i => i.classList.remove('active'));
    } else {
      currentTag = tag;
      document.querySelectorAll('.tag-item').forEach(i => i.classList.remove('active'));
      document.querySelectorAll('.tag-item').forEach(i => {
        if (i.textContent.trim() === tag) i.classList.add('active');
      });
    }
    applyFilters();
  }

  function filterByText() {
    applyFilters();
  }

  function applyFilters() {
    const searchEl = document.getElementById('artSearch');
    const searchText = searchEl ? searchEl.value.toLowerCase() : '';
    const rows = document.querySelectorAll('.lecture-row');

    rows.forEach(row => {
      const title = row.querySelector('a').innerText.toLowerCase();
      const tags = row.dataset.tags.split(',');

      const matchesText = title.includes(searchText) || tags.some(t => t.toLowerCase().includes(searchText));
      const matchesTag = !currentTag || tags.includes(currentTag);

      if (matchesText && matchesTag) {
        row.classList.remove('hidden-row');
      } else {
        row.classList.add('hidden-row');
      }
    });
  }

  function clearFilters() {
    currentTag = null;
    const searchEl = document.getElementById('artSearch');
    if (searchEl) searchEl.value = '';
    document.querySelectorAll('.tag-item').forEach(i => i.classList.remove('active'));
    document.querySelectorAll('.lecture-row').forEach(row => row.classList.remove('hidden-row'));
  }

  (function sortLectures() {
    const tbody = document.getElementById('lectureBody');
    if (!tbody) return;
    const rows = Array.from(tbody.querySelectorAll('tr.lecture-row'));
    rows.sort((a, b) => {
      const dateA = a.dataset.date || '';
      const dateB = b.dataset.date || '';
      if (dateA && dateB) return dateA.localeCompare(dateB);
      if (dateA) return -1;
      if (dateB) return 1;
      return parseInt(a.dataset.num) - parseInt(b.dataset.num);
    });
    rows.forEach(row => tbody.appendChild(row));
  })();
</script>
