.PHONY: serve build deploy install

serve:
	jekyll serve --livereload

build:
	JEKYLL_ENV=production jekyll build

deploy:
	yarn deploy


install:
	bundle install
	yarn add postcss@latest tailwindcss@latest autoprefixer@latest cssnano@latest -D
